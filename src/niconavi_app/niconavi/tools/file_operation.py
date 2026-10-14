# %%
import os
import yaml  # type: ignore
from typing import Dict, Any, Tuple, List

def extract_numbers_from_filename(file_name: str) -> Tuple[float, float]:
    """
    Extracts the numeric parts 'xxxx' and 'yyyy' from a file name of the form 'abcd_xxxx_yyyy.ext'.
    The numbers are parsed according to specific rules:

    - '100'    -> 100    (integer)
    - '123'    -> 123    (integer)
    - '12-33'  -> 12.33  (float)
    - '99-00'  -> 99.00  (float)
    - '12a39'  -> Error  (contains letters)
    - '0121'   -> Error  (leading zeros not allowed)
    - '12-33-4'-> Error  (too many dashes)

    Parameters:
        file_name (str): The file name to process.

    Returns:
        Tuple[float, float]: A tuple containing the two extracted numbers.

    Raises:
        ValueError: If the file name does not conform to the expected format or contains invalid numbers.
    """
    # Remove the file extension
    base_name, ext = os.path.splitext(file_name)

    # Split the base name into parts
    parts = base_name.split("_")
    if len(parts) < 3:
        raise ValueError(
            "File name does not contain enough parts separated by underscores."
        )

    # Extract the 'xxxx' and 'yyyy' parts
    xxxx_str = parts[-2]
    yyyy_str = parts[-1]

    # Parse the numbers according to the specified rules
    xxxx = parse_number(xxxx_str)
    yyyy = parse_number(yyyy_str)

    return xxxx, yyyy


def parse_number(s: str) -> float:
    """
    Parses a string into a number according to specific rules:

    - If the string is all digits, it is converted to an integer.
    - If the string is in 'xx-yy' format, it is converted to a float 'xx.yy'.
    - Leading zeros are not allowed in the integer part (unless the number is zero).
    - The string must not contain any letters.
    - Only one dash is allowed; more than one dash is invalid.

    Parameters:
        s (str): The string to parse.

    Returns:
        float: The parsed number.

    Raises:
        ValueError: If the string does not conform to the expected number format.
    """
    # Check for letters in the string
    if any(c.isalpha() for c in s):
        raise ValueError(f"Invalid number format (contains letters): '{s}'")

    # Count dashes to determine the format
    num_dashes = s.count("-")
    if num_dashes == 0:
        # Integer format
        if not s.isdigit():
            raise ValueError(f"Invalid integer format: '{s}'")
        if s.startswith("0") and len(s) > 1:
            raise ValueError(f"Leading zeros are not allowed: '{s}'")
        return int(s)
    elif num_dashes == 1:
        # Decimal format
        integer_part, decimal_part = s.split("-")
        if not integer_part.isdigit() or not decimal_part.isdigit():
            raise ValueError(f"Invalid decimal format: '{s}'")
        if integer_part.startswith("0") and len(integer_part) > 1:
            raise ValueError(f"Leading zeros are not allowed: '{s}'")
        # Construct the float number
        float_str = f"{integer_part}.{decimal_part}"
        return float(float_str)
    else:
        # Invalid format with too many dashes
        raise ValueError(f"Invalid number format (too many dashes): '{s}'")


def make_dir(root_path: str, dirname: str) -> None:
    """
    Creates a directory named 'dirname' inside 'root_path'.
    If the directory already exists, it does nothing.

    Parameters:
        root_path (str): The path of the root directory.
        dirname (str): The name of the directory to create.

    Returns:
        None
    """
    # Create the full directory path
    dir_path = os.path.join(root_path, dirname)

    # Create the directory (does nothing if it already exists)
    os.makedirs(dir_path, exist_ok=True)


def read_parameter_yaml(
    path_of_file: str, file_name: str = "_parameter.yaml"
) -> Dict[str, Any]:
    """
    Reads the parameter YAML file from the parameter directory.

    If the file exists, returns its data.
    If it does not exist, returns an empty dictionary.

    Parameters:
        file_name (str): The name of the YAML file to read.

    Returns:
        Dict[str, Any]: The contents of the YAML file as a dictionary.
    """
    file_path = os.path.join(path_of_file, file_name)
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
            return data if data else {}
    else:
        return {}