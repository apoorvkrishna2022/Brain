import os
from typing import Dict, Any, Optional, List
from pydantic import Field

from .base import BaseTool, ToolInput

class FileReadInput(ToolInput):
    """Input schema for file read tool"""
    file_path: str = Field(..., description="Path to the file to read")
    start_line: int = Field(default=0, description="Line to start reading from (0-indexed)")
    num_lines: Optional[int] = Field(default=None, description="Number of lines to read (None for all)")

class FileWriteInput(ToolInput):
    """Input schema for file write tool"""
    file_path: str = Field(..., description="Path to the file to write")
    content: str = Field(..., description="Content to write to the file")
    append: bool = Field(default=False, description="Whether to append to the file instead of overwriting")

class FileReadTool(BaseTool):
    """Tool for reading files"""
    name = "file_read"
    description = "Read the contents of a file from the filesystem"
    input_schema = FileReadInput
    
    def _run(self, file_path: str, start_line: int = 0, num_lines: Optional[int] = None) -> str:
        """Read content from a file"""
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                
                # Apply line range if specified
                if start_line >= len(lines):
                    return f"Error: Start line {start_line} is out of range (file has {len(lines)} lines)"
                
                if num_lines is None:
                    selected_lines = lines[start_line:]
                else:
                    selected_lines = lines[start_line:start_line + num_lines]
                
                return ''.join(selected_lines)
        except FileNotFoundError:
            return f"Error: File not found: {file_path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

class FileWriteTool(BaseTool):
    """Tool for writing to files"""
    name = "file_write"
    description = "Write content to a file on the filesystem"
    input_schema = FileWriteInput
    
    def _run(self, file_path: str, content: str, append: bool = False) -> str:
        """Write content to a file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Write to file
            mode = 'a' if append else 'w'
            with open(file_path, mode) as file:
                file.write(content)
                
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing to file: {str(e)}" 