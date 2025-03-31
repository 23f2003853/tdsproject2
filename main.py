import os
import io
import zipfile
import json
import pandas as pd
import PyPDF2
import tempfile
import shutil
import base64
import mimetypes

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel # For response model

from docx import Document
from PIL import Image, ImageDraw
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# --- Configuration ---

# IMPORTANT: Set your Anthropic API Key as an environment variable
# export ANTHROPIC_API_KEY="your_real_api_key_here"
# os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY" # Use environment variable

# Check for API Key
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set. Please set it before running.")

# --- Constants ---
TEMP_DIR_PREFIX = "fastapi_claude_temp_"
MAX_FILE_SIZE_MB = 100 # Limit upload size (adjust as needed)
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_CONTENT_LENGTH_FOR_PROMPT = 75000 # Limit text content sent to LLM
TARGET_COMPRESSION_SIZE_BYTES = 3000 # For compress_image tool

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".xlsx", ".xls", ".pdf", ".json", ".docx"}
SUPPORTED_ZIP_EXTENSIONS = {".zip"}

# --- FastAPI App Initialization ---
app = FastAPI()

class AnswerResponse(BaseModel):
    answer: str

# --- Utility Functions (Adapted/Combined) ---

def get_file_extension(filename: str) -> str:
    """Safely gets the lowercase file extension."""
    return os.path.splitext(filename)[1].lower()

def is_image_file(filename: str) -> bool:
    """Check if filename has a supported image extension."""
    return get_file_extension(filename) in SUPPORTED_IMAGE_EXTENSIONS

def is_text_or_doc_file(filename: str) -> bool:
    """Check if filename has a supported text/doc extension."""
    return get_file_extension(filename) in SUPPORTED_TEXT_EXTENSIONS

def is_zip_file(filename: str) -> bool:
    """Check if filename has a supported zip extension."""
    return get_file_extension(filename) in SUPPORTED_ZIP_EXTENSIONS

async def save_upload_file_tmp(upload_file: UploadFile) -> str:
    """Saves UploadFile to a temporary file and returns the path."""
    if not upload_file.filename:
         raise HTTPException(status_code=400, detail="File has no name.")

    # Basic size check
    # Note: Reading the whole file into memory first for size check isn't ideal for large files.
    # A more robust solution would stream and count, but this is simpler for now.
    contents = await upload_file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File size exceeds {MAX_FILE_SIZE_MB} MB limit.")

    # Create a temporary directory to store the file
    temp_dir = tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX)
    # Use the original filename within the temporary directory
    temp_file_path = os.path.join(temp_dir, upload_file.filename)

    try:
        with open(temp_file_path, "wb") as f:
            f.write(contents) # Write the already read contents
    except Exception as e:
        shutil.rmtree(temp_dir) # Clean up directory on write error
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {e}")

    return temp_file_path # Return the full path to the saved file

def cleanup_temp_file(filepath: str):
    """Removes the temporary file and its containing directory."""
    if filepath and os.path.exists(filepath):
        temp_dir = os.path.dirname(filepath)
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Error cleaning up temporary directory {temp_dir}: {e}")

def image_to_base64(image_path: str) -> str | None:
    """Converts an image file to a base64 encoded data URI string."""
    if not os.path.exists(image_path):
        print(f"Error [image_to_base64]: Image file not found at '{image_path}'")
        return None
    try:
        with Image.open(image_path) as img:
            # Determine format
            format = img.format
            if not format: # If format isn't read from metadata, guess from extension
                ext = os.path.splitext(image_path)[1].lower()
                if ext == ".png": format = "PNG"
                elif ext in [".jpg", ".jpeg"]: format = "JPEG"
                elif ext == ".gif": format = "GIF"
                elif ext == ".webp": format = "WEBP"
                else: # Fallback if extension is unknown/missing but PIL could open it
                    print(f"Warning [image_to_base64]: Could not determine format for {image_path}. Trying to save as PNG.")
                    buffer = io.BytesIO()
                    img.save(buffer, format="PNG")
                    encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    return f"data:image/png;base64,{encoded_string}"

            format_lower = format.lower()
            if format_lower == "jpg": format_lower = "jpeg" # Standardize common variations

            # Read raw bytes and encode
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/{format_lower};base64,{encoded_string}"
    except Exception as e:
        print(f"Error [image_to_base64]: Failed converting '{image_path}': {e}")
        return None

# --- File Reader Logic (Adapted from Snippet 1) ---

class FileReader:
    """Class to read various file formats and extract their content from a given filepath"""

    @staticmethod
    def read_text_file(filepath):
        """Read text or markdown files"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(filepath, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                 return f"Error reading text file with multiple encodings: {str(e)}"
        except Exception as e:
            return f"Error reading text file: {str(e)}"

    @staticmethod
    def read_csv_file(filepath):
        """Read CSV files and convert to string"""
        try:
            # Try common encodings
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding='latin-1')
            return df.to_string()
        except Exception as e:
            return f"Error reading CSV: {str(e)}"

    @staticmethod
    def read_excel_file(filepath):
        """Read Excel files and convert to string"""
        try:
            # Reading all sheets might be necessary depending on the use case
            xls = pd.ExcelFile(filepath)
            all_sheets_content = []
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                all_sheets_content.append(f"--- Sheet: {sheet_name} ---\n{df.to_string()}")
            return "\n\n".join(all_sheets_content)
        except Exception as e:
            return f"Error reading Excel: {str(e)}"

    @staticmethod
    def read_pdf_file(filepath):
        """Read PDF files"""
        try:
            text = ""
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    try: # Handle potential errors extracting text from a specific page
                        page_text = pdf_reader.pages[page_num].extract_text()
                        if page_text:
                           text += page_text + "\n"
                    except Exception as page_e:
                         print(f"Warning: Could not extract text from PDF page {page_num + 1}: {page_e}")
                         text += f"[Could not extract text from page {page_num + 1}]\n"

            if not text and len(pdf_reader.pages) > 0:
                 return "[Warning: PDF detected, but no text could be extracted. It might be an image-based PDF.]"
            return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"

    @staticmethod
    def read_json_file(filepath):
        """Read JSON files"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return json.dumps(data, indent=2)
        except Exception as e:
            return f"Error reading JSON: {str(e)}"

    @staticmethod
    def read_docx_file(filepath):
        """Read DOCX files"""
        try:
            doc = Document(filepath)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            return f"Error reading DOCX: {str(e)}"

    @staticmethod
    def process_file_from_zip(file_content, filename, temp_dir_for_zip):
        """Process a file's byte content extracted from a zip based on its extension"""
        extension = get_file_extension(filename)
        # We need to write the bytes to a temporary file *within* the zip's temp dir
        # because some readers (PDF, DOCX, Excel) expect a file path.
        temp_file_path = os.path.join(temp_dir_for_zip, f"zip_extract_{os.path.basename(filename)}")

        try:
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_content)

            if extension in ['.txt', '.md']:
                # Read back the written file for consistent encoding handling
                return FileReader.read_text_file(temp_file_path)
            elif extension == '.csv':
                return FileReader.read_csv_file(temp_file_path)
            elif extension in ['.xlsx', '.xls']:
                return FileReader.read_excel_file(temp_file_path)
            elif extension == '.pdf':
                 # PyPDF2 needs path, so we use the temp path
                return FileReader.read_pdf_file(temp_file_path)
            elif extension == '.json':
                # JSON can often be decoded directly, but reading file handles encoding better
                return FileReader.read_json_file(temp_file_path)
            elif extension == '.docx':
                # python-docx needs path
                return FileReader.read_docx_file(temp_file_path)
            elif is_image_file(filename):
                # Don't process image content for text QA, just note its presence
                 return f"[Image file: {filename} - Content not extracted for text QA]"
            else:
                return f"[Unsupported file type in ZIP: {filename} ({extension})]"
        except Exception as e:
            return f"Error processing {filename} from ZIP: {str(e)}"
        finally:
             # Clean up the individual extracted file temp
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as rm_e:
                     print(f"Warning: Could not remove temp extracted file {temp_file_path}: {rm_e}")


    @staticmethod
    def read_zip_file(filepath):
        """Extract and read files from a zip file, returning combined text content"""
        contents = {}
        # Create a dedicated subdirectory within the main temp dir for extracted files
        zip_extract_dir = filepath + "_extracted"
        os.makedirs(zip_extract_dir, exist_ok=True)

        try:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    # Ignore directories and MacOS resource forks
                    if file_info.is_dir() or file_info.filename.startswith('__MACOSX/'):
                        continue

                    filename = os.path.basename(file_info.filename) # Use basename to avoid path issues
                    if not filename: # Skip if filename is empty after basename (e.g., just a directory separator)
                        continue

                    try:
                        with zip_ref.open(file_info) as file:
                            file_content_bytes = file.read()
                            # Pass the dedicated dir for temporary storage of extracted files
                            contents[file_info.filename] = FileReader.process_file_from_zip(
                                file_content_bytes, filename, zip_extract_dir
                             )
                    except Exception as extract_e:
                         print(f"Error opening/reading {file_info.filename} in zip: {extract_e}")
                         contents[file_info.filename] = f"[Error reading file in zip: {extract_e}]"

            # Combine all file contents with their filenames
            combined_content = f"--- Contents of ZIP file: {os.path.basename(filepath)} ---\n"
            if not contents:
                combined_content += "[ZIP file is empty or contains only directories/unreadable files]"
            else:
                for filename, content in contents.items():
                     # Truncate content *per file* if necessary before combining
                     if len(content) > MAX_CONTENT_LENGTH_FOR_PROMPT:
                         content = content[:MAX_CONTENT_LENGTH_FOR_PROMPT] + f"\n\n[Note: Content truncated. Original length: {len(content)} characters]"
                     combined_content += f"\n\n--- File: {filename} ---\n{content}"

            return combined_content
        except zipfile.BadZipFile:
             return f"Error: Invalid or corrupted ZIP file '{os.path.basename(filepath)}'."
        except Exception as e:
            return f"Error reading zip file: {str(e)}"
        finally:
            # Clean up the directory used for extracted files
             if os.path.exists(zip_extract_dir):
                 try:
                     shutil.rmtree(zip_extract_dir)
                 except Exception as clean_zip_e:
                     print(f"Warning: Could not clean up zip extract directory {zip_extract_dir}: {clean_zip_e}")


    @staticmethod
    def read_file(filepath):
        """Read a file based on its type, returning its text content or an error string."""
        if not os.path.exists(filepath):
            return f"Error: File '{os.path.basename(filepath)}' does not exist at expected temporary path."

        extension = get_file_extension(filepath)

        try:
            if extension in ['.txt', '.md']:
                return FileReader.read_text_file(filepath)
            elif extension == '.csv':
                return FileReader.read_csv_file(filepath)
            elif extension in ['.xlsx', '.xls']:
                return FileReader.read_excel_file(filepath)
            elif extension == '.pdf':
                return FileReader.read_pdf_file(filepath)
            elif extension == '.json':
                return FileReader.read_json_file(filepath)
            elif extension == '.docx':
                return FileReader.read_docx_file(filepath)
            elif extension == '.zip':
                return FileReader.read_zip_file(filepath)
            # Image files are handled separately by tools, this function focuses on text content
            elif is_image_file(filepath):
                 return f"[Image file detected: {os.path.basename(filepath)} - Use image-specific questions/tools]"
            else:
                # Try reading as plain text as a last resort for unknown types
                 print(f"Warning: Unsupported file format '{extension}'. Attempting to read as text.")
                 content = FileReader.read_text_file(filepath)
                 if content.startswith("Error"):
                     return f"Error: Unsupported file format '{extension}' and could not read as text."
                 else:
                     return content + f"\n[Note: Read as plain text, format '{extension}' unknown]"
        except Exception as e:
            return f"Error reading file '{os.path.basename(filepath)}': {str(e)}"

# --- Langchain Tools (Adapted from Snippet 2) ---

@tool
def get_openai_format(text_prompt: str, image_fname: str = "") -> str:
    """
    Generates an OpenAI API compatible JSON payload for multimodal input.
    Uses the provided text prompt and optionally an image filename (full path expected).
    The tool will attempt to find and encode the image file specified by image_fname.

    Args:
        text_prompt: The user's text prompt.
        image_fname: Optional full path to the image file to include. Leave empty if no image.

    Returns:
        A JSON string representing the OpenAI request body, or an error message.
    """
    content = [{"type": "text", "text": text_prompt}]
    image_base64 = None # Initialize

    if image_fname and image_fname.strip():
        image_path = image_fname.strip()
        print(f"Tool '{get_openai_format.name}': Received image path '{image_path}'. Attempting to encode.")
        # *** Internal Base64 Conversion ***
        if not os.path.exists(image_path):
             error_msg = f"Error: Tool '{get_openai_format.name}' could not find the image file at path '{image_path}'."
             print(error_msg)
             return json.dumps({"error": error_msg}) # Return error JSON

        image_base64 = image_to_base64(image_path)
        if image_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_base64}
            })
            print(f"Tool '{get_openai_format.name}': Successfully encoded and added image.")
        else:
            # Handle case where encoding failed but file existed
            error_msg = f"Error: Tool '{get_openai_format.name}' failed to encode image from path '{image_path}'. Proceeding without image data in payload."
            print(error_msg)
            content.append({"type": "text", "text": f"(Note: Failed to load/encode image from '{os.path.basename(image_path)}')"}) # Add note

    # Construct final payload
    payload = {
        "model": "gpt-4o", # Or gpt-4-turbo, gpt-4o-mini etc.
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 1000 # Example: add other OpenAI params if needed
    }
    return json.dumps(payload, indent=2)

@tool
def compress_image(input_image_path: str, output_image_path: str) -> str:
    """
    Attempts to save an image losslessly as PNG under a specific size limit (TARGET_COMPRESSION_SIZE_BYTES).
    Requires full input and output file paths.

    Args:
        input_image_path: The full path to the source image file.
        output_image_path: The full path where the compressed PNG should be saved
                           if the size constraint is met. The directory will be created if needed.

    Returns:
        A message indicating success (including final size) or failure.
    """
    if not input_image_path or not output_image_path:
        return "Error: Both input_image_path and output_image_path must be provided."
    if not os.path.exists(input_image_path):
        return f"Error: Input file not found at '{input_image_path}'"
    try:
        with Image.open(input_image_path) as img:
            img_byte_arr = io.BytesIO()
            # Use high compression settings for PNG
            img.save(img_byte_arr, format="PNG", optimize=True, compress_level=9)
            compressed_size = img_byte_arr.tell()

            if compressed_size < TARGET_COMPRESSION_SIZE_BYTES:
                output_dir = os.path.dirname(output_image_path)
                if output_dir: os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists
                with open(output_image_path, "wb") as f_out:
                    f_out.write(img_byte_arr.getvalue())
                return (f"Success: Image losslessly compressed to {compressed_size} bytes "
                        f"(<{TARGET_COMPRESSION_SIZE_BYTES}) and saved as '{os.path.basename(output_image_path)}'. "
                        f"Full path: '{output_image_path}'") # Return path for clarity if needed elsewhere
            else:
                return (f"Failure: Cannot compress losslessly below {TARGET_COMPRESSION_SIZE_BYTES} bytes. "
                        f"Resulting PNG size would be {compressed_size} bytes. "
                        f"No file was created at '{output_image_path}'.")
    except FileNotFoundError:
        return f"Error: Input file not found at '{input_image_path}'."
    except Exception as e:
        # Provide more specific error if PIL fails to open/save
        return f"Error processing image '{os.path.basename(input_image_path)}' to '{os.path.basename(output_image_path)}': {e}"


@tool
def answer_image_question(user_question: str, image_fname: str) -> str:
    """
    Answers a user's question based on the content of an image file.
    Requires the user question and the full path to the image file (image_fname).
    This tool internally loads the image and uses a multimodal model (Claude 3) for analysis.

    Args:
        user_question: The question asked by the user about the image.
        image_fname: The full path to the image file to analyze.

    Returns:
        An answer to the question based on the image, or an error message.
    """
    if not image_fname or not image_fname.strip():
        return "Error: An image file path (image_fname) is required to answer the question."
    if not user_question or not user_question.strip():
         return "Error: A user question is required."

    image_path = image_fname.strip()
    print(f"Tool '{answer_image_question.name}': Received question '{user_question}' for image path '{image_path}'.")

    if not os.path.exists(image_path):
         error_msg = f"Error: Tool '{answer_image_question.name}' could not find the image file at path '{image_path}'."
         print(error_msg)
         return error_msg # Return error to LLM

    # *** Internal Base64 Conversion ***
    image_base64 = image_to_base64(image_path)

    if not image_base64:
        # Handle case where image loading/encoding failed
        error_msg = f"Error: Tool '{answer_image_question.name}' failed to load or encode image from path '{image_path}'."
        print(error_msg)
        return error_msg # Return error to LLM

    # --- Actual Vision Model Call ---
    print(f"Tool '{answer_image_question.name}': Successfully encoded image. Querying Claude 3...")
    try:
        # Use the base LLM instance (already initialized)
        # Construct the multimodal message for Claude
        vision_message = HumanMessage(
            content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_base64.split(";")[0].split(":")[1], # e.g., "image/png"
                        "data": image_base64.split(",")[1],
                    },
                },
                {
                    "type": "text",
                    "text": user_question
                },
            ]
        )
        # Invoke the LLM directly with the structured message
        response = llm2.invoke([vision_message])
        print(f"Tool '{answer_image_question.name}': Received response from Claude.")
        # Assuming the response object has a 'content' attribute with the text answer
        return response.content if hasattr(response, 'content') else "Error: Could not parse response from Claude."

    except Exception as e:
        error_msg = f"Error: Tool '{answer_image_question.name}' encountered an error calling the vision model: {e}"
        print(error_msg)
        return error_msg
    # --- End Vision Model Call ---

# --- LLM Initialization and Tool Binding ---

tools = [
    get_openai_format,
    compress_image,
    answer_image_question
]

# Initialize base LLM (can be used for text-only or direct vision calls within tools)
llm1 = ChatAnthropic(
    model="claude-3-7-sonnet-20250219", 
    max_tokens=3000,
    thinking={"type": "enabled", "budget_tokens": 2000},
    anthropic_api_key=ANTHROPIC_API_KEY,
)

llm2 = ChatAnthropic(
    model="claude-3-7-sonnet-20250219", 
    max_tokens=3000,
    anthropic_api_key=ANTHROPIC_API_KEY,
)

# Bind tools to the LLM - this allows the LLM to decide *which* tool to call
llm_with_tools = llm2.bind_tools(tools)


# --- FastAPI Endpoint ---

@app.post("/api/", response_model=AnswerResponse)
async def process_question_with_file(
    question: str = Form(...),
    file: UploadFile | None = File(None) # Make file optional
):
    """
    Processes a question, optionally using an uploaded file for context.
    Handles text files, zip archives, and image files (using tools).
    """
    temp_file_path = None
    llm_response = None
    final_answer = "Error: Could not determine answer." # Default error

    try:
        if file:
            # --- File Handling ---
            if not file.filename:
                raise HTTPException(status_code=400, detail="Received file is missing a filename.")

            # Save the uploaded file temporarily
            temp_file_path = await save_upload_file_tmp(file)
            print(f"File '{file.filename}' saved temporarily to '{temp_file_path}'")
            file_extension = get_file_extension(file.filename)

            # --- Logic Branching based on File Type ---

            # Branch 1: Image File - Use Tool-based approach
            if is_image_file(file.filename):
                print(f"Detected Image file: {file.filename}")
                # The prompt needs to guide the LLM to use a tool AND provide the filename
                # Modify the question slightly to include context about the file path
                # Note: Claude might still need explicit instruction like "Using the image at path..."
                prompt_for_llm = (
                    f"User Question: {question}\n\n"
                    f"Context: An image file has been provided at the path: '{temp_file_path}'. "
                    f"Please use the available tools to answer the question or perform the requested action regarding this image."
                    # Example guiding phrases (optional, LLM might infer):
                    # f" If the question is about the image content, use 'answer_image_question'."
                    # f" If the question is about compression, use 'compress_image'."
                    # f" If the question is about generating API formats, use 'get_openai_format'."
                )
                print(f"Invoking LLM with tools for image. Prompt includes reference to: {temp_file_path}")

                # Invoke LLM - it should decide to call a tool
                initial_response = llm_with_tools.invoke([HumanMessage(content=prompt_for_llm)])

                # Check if the LLM decided to call a tool
                if hasattr(initial_response, 'tool_calls') and initial_response.tool_calls:
                    print(f"LLM decided to call tools: {initial_response.tool_calls}")
                    tool_results = []
                    # Execute the tool calls
                    for tool_call in initial_response.tool_calls:
                         tool_name = tool_call['name']
                         tool_args = tool_call['args']
                         tool_id = tool_call['id'] # Important for ToolMessage
                         print(f"Executing tool: {tool_name} with args: {tool_args}")

                         # Find the corresponding tool function and execute it
                         tool_function = next((t for t in tools if t.name == tool_name), None)
                         if tool_function:
                             try:
                                 # Ensure filename argument uses the temp path if required by the tool
                                 if 'image_fname' in tool_args and tool_name in [answer_image_question.name, get_openai_format.name]:
                                     tool_args['image_fname'] = temp_file_path # Override with actual temp path
                                 if 'input_image_path' in tool_args and tool_name == compress_image.name:
                                      tool_args['input_image_path'] = temp_file_path # Override input path
                                      # We might need to generate a temp output path for compression tool
                                      if 'output_image_path' not in tool_args or not tool_args['output_image_path']:
                                           base, ext = os.path.splitext(temp_file_path)
                                           tool_args['output_image_path'] = f"{base}_compressed.png"
                                           print(f"Generated temporary output path for compress_image: {tool_args['output_image_path']}")


                                 result = tool_function.invoke(tool_args)
                                 print(f"Tool {tool_name} result: {result}")
                                 tool_results.append(ToolMessage(content=str(result), tool_call_id=tool_id))
                             except Exception as e:
                                 print(f"Error executing tool {tool_name}: {e}")
                                 # Report error back to the LLM via ToolMessage
                                 tool_results.append(ToolMessage(content=f"Error executing tool {tool_name}: {e}", tool_call_id=tool_id))
                         else:
                              print(f"Error: Tool '{tool_name}' not found.")
                              tool_results.append(ToolMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_id))

                    # Send tool results back to the LLM for final answer generation
                    print("Sending tool results back to LLM for final answer...")
                    final_llm_response = llm_with_tools.invoke([
                         HumanMessage(content=prompt_for_llm), # Original request
                         initial_response, # LLM's first response (tool calls)
                         *tool_results # Results from executed tools
                     ])
                    # llm_response = final_llm_response # Use the final response after tool execution
                    try:
                        llm_response = final_llm_response.content + str(result)
                    except:
                        llm_response = final_llm_response.content


                else:
                    # LLM didn't call a tool, use its direct response (might be fallback or error)
                    print("LLM did not call a tool for the image query. Using direct response.")
                    llm_response = initial_response

            # Branch 2: Text/Document/Zip File - Read content and pass to LLM
            elif is_text_or_doc_file(file.filename) or is_zip_file(file.filename):
                print(f"Detected Text/Doc/Zip file: {file.filename}")
                file_content = FileReader.read_file(temp_file_path)

                if file_content.startswith("Error:") or file_content.startswith("[Warning:"):
                     print(f"Issue reading file content: {file_content}")
                     # Decide if we should proceed with just the question or return the error
                     # Option 1: Return the file reading error
                     # raise HTTPException(status_code=400, detail=f"Error processing uploaded file: {file_content}")
                     # Option 2: Ask question without content, but add a note
                     prompt_for_llm = f"{question}\n\n[Note: A file '{file.filename}' was uploaded but could not be fully processed: {file_content}]"
                     prompt_for_llm += """
Instructions:
1. Provide ONLY the exact answer to the question, without any explanations or extra text.
2. The answer should be ready to be directly entered into the assignment form.
3. If the question asks for values from a CSV file or data, extract those specific values.
4. If the answer is a number, provide just the number.
5. If the answer is text, provide just the text.
6. Do not include any explanations, citations, or your thought process.

Your answer should be extremely concise and exactly match what is required for the assignment.
"""
                     print("Invoking LLM with question and note about file processing issue.")
                     llm_response = llm1.invoke([HumanMessage(content=prompt_for_llm)])

                else:
                    # Truncate content if too long BEFORE sending to LLM
                    original_length = len(file_content)
                    if original_length > MAX_CONTENT_LENGTH_FOR_PROMPT:
                        file_content = file_content[:MAX_CONTENT_LENGTH_FOR_PROMPT] + \
                                        f"\n\n[Note: Content truncated. Original length: {original_length} characters]"
                        print(f"Truncated file content from {original_length} to {len(file_content)} characters.")

                    prompt_for_llm = f"""Based on the content of the file '{file.filename}' provided below, please answer the following question.

Question:
{question}

File Content:
---BEGIN FILE CONTENT---
{file_content}
---END FILE CONTENT---
"""
                    print(f"Invoking LLM with question and content from: {file.filename}")
                    llm_response = llm1.invoke([HumanMessage(content=prompt_for_llm)])

            # Branch 3: Unsupported File Type
            else:
                print(f"Unsupported file type: {file.filename} (Extension: {file_extension})")
                raise HTTPException(
                    status_code=415, # Unsupported Media Type
                    detail=f"Unsupported file type: '{file_extension}'. Supported types are text-based (txt, md, csv, pdf, json, docx, xlsx), zip archives, and images (png, jpg, gif, webp)."
                )

        else:
            # --- No File Provided ---
            print("No file provided. Processing question directly.")
            prompt_for_llm = f"""
Question:
{question}

Instructions:
1. Provide ONLY the exact answer to the question, without any explanations or extra text.
2. The answer should be ready to be directly entered into the assignment form.
"""
            llm_response = llm1.invoke([HumanMessage(content=prompt_for_llm)])

        # --- Process Final LLM Response ---
        if llm_response and hasattr(llm_response, 'content'):
            try:
                final_answer = llm_response.content[-1]["text"]
            except:
                final_answer = llm_response.content
            # Basic check if the answer looks like an error message returned by tools/LLM
            if "error:" in final_answer.lower() or "failed:" in final_answer.lower():
                 print(f"Warning: LLM answer might contain an error message: {final_answer[:100]}...")
            # Clean up potential placeholder/refusal text if possible
            # (This is simple, more robust checks might be needed)
            if "placeholder response:" in final_answer.lower():
                print("Removing placeholder text from final answer.")
                final_answer = final_answer.split("Placeholder response:")[-1].strip()
            if "analysis is not implemented yet" in final_answer.lower():
                 final_answer = "[Analysis functionality is not fully implemented in the tool]"


        elif llm_response:
             # Handle cases where response structure might be different
             print(f"Warning: LLM response format might be unexpected: {type(llm_response)}")
             final_answer = str(llm_response) # Fallback to string representation

        else:
            # This case should ideally not be reached if LLM call succeeds/fails gracefully
            print("Error: No valid response received from LLM.")
            final_answer = "Error: Failed to get a response from the language model."


    except HTTPException as e:
         # FastAPI HTTPExceptions are re-raised to be handled by FastAPI
         raise e
    except Exception as e:
        # Catch-all for other unexpected errors during processing
        print(f"An unexpected error occurred: {e}")
        # Optionally log the full traceback here
        import traceback
        traceback.print_exc()
        # Return a generic server error
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

    finally:
        # --- Cleanup ---
        if temp_file_path:
            cleanup_temp_file(temp_file_path)
            # Clean up any compressed file if created by the tool
            compressed_path_guess = temp_file_path.replace(get_file_extension(temp_file_path), "_compressed.png")
            if os.path.exists(compressed_path_guess):
                 cleanup_temp_file(compressed_path_guess) # This will remove the parent dir again, but shutil.rmtree handles it gracefully


    # Return the final answer in the required JSON format
    print(f"Final Answer: {final_answer}")
    return AnswerResponse(answer=final_answer)

# --- Example of how to run this with uvicorn ---
# Save the code as main.py
# Run in terminal: uvicorn main:app --reload --host 0.0.0.0 --port 8000
#
# Then you can use curl:
# curl -X POST "http://localhost:8000/api/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "question=What is this file about?" -F "file=@./your_document.pdf"
# curl -X POST "http://localhost:8000/api/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "question=Describe this image" -F "file=@./your_image.png"
# curl -X POST "http://localhost:8000/api/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "question=What is the capital of France?"
# curl -X POST "http://localhost:8000/api/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "question=Download and unzip file example.zip which has a single extract.csv file inside. What is the value in the "answer" column of the CSV file?" -F "file=@./example.zip"

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    print("Ensure ANTHROPIC_API_KEY environment variable is set.")
    print("API Endpoint available at http://localhost:8000/api/")
    print("Docs available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)