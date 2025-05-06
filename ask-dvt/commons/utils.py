import json, re, colorsys, time, uuid
from typing import List
from google.cloud import storage
from google.api_core import exceptions
from langchain_core.messages import ToolMessage, BaseMessage, HumanMessage, AIMessage

class Utilities:
    @staticmethod
    def extract_json_from_string(input_string):
        """
        Extracts a JSON dictionary from a string that may contain additional characters.
        Args:
            input_string (str): The input string to parse.
        Returns:
            dict or None: A dictionary if valid JSON is found, or None if not.
        """
        try:
            # Find the first and last curly braces
            start_index = input_string.find('{')
            end_index = input_string.rfind('}')

            if start_index == -1 or end_index == -1:
                return None  # No curly braces found

            json_string = input_string[start_index:end_index + 1]

            # Attempt to parse the extracted string as JSON
            return json.loads(json_string)

        except json.JSONDecodeError:
            # Attempt to fix common errors and re-try.
            # Remove any non-printable or control characters.
            cleaned_string = re.sub(r'[^\x20-\x7E]+', '', json_string)

            # Attempt to fix single quotes to double quotes, if needed.
            cleaned_string = re.sub(r"'([^']+)'", r'"\1"', cleaned_string)

            try:
                return json.loads(cleaned_string)
            except json.JSONDecodeError:
                return None  # Could not parse as JSON

        except Exception:
            return None

    @staticmethod
    def extract_json_list_of_dicts(text):
        """
        Extracts JSON from text and always returns a list of dictionaries.

        Args:
            text (str): The input text containing the JSON data.

        Returns:
            list or None: A list of dictionaries, or None if no valid JSON is found.
        """
        json_pattern = r'```json\s*({.*?}|\[.*?\])\s*```'
        try:
            match = re.search(json_pattern, text, re.DOTALL)
        except:
            return None

        if match:
            try:
                json_str = match.group(1)
                data = json.loads(json_str)

                if isinstance(data, dict):
                    return [data]  # Wrap single dictionary in a list
                elif isinstance(data, list):
                    # Ensure all elements in the list are dictionaries
                    if all(isinstance(item, dict) for item in data):
                        return data
                    else:
                        return None # return None if list contains non-dictionary items
                else:
                    return None  # Return None if JSON is neither a dict nor a list

            except json.JSONDecodeError:
                return None
        else:
            return None
        

    @staticmethod
    def generate_hex_colors(num_colors):
        """
        Generates a list of hex color codes given the number of colors.

        Args:
            num_colors: The number of hex color codes to generate.

        Returns:
            A list of hex color codes as strings.
        """
        import colorsys
        if num_colors <= 0:
            return []

        hex_colors = []
        for i in range(num_colors):
            hue = i / num_colors  # Evenly distribute hues
            saturation = 0.8       # Adjust saturation as needed
            value = 0.9            # Adjust value (brightness) as needed

            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
            hex_colors.append(hex_color)

        return hex_colors
    
    @staticmethod
    def generate_unique_number():
        """Generates a highly unique number using timestamp and UUID."""
        
        # Get the current time in nanoseconds (or as close as possible).
        timestamp_ns = time.time_ns()

        # Generate a UUID (Universally Unique Identifier).
        unique_id = uuid.uuid4()

        # Combine the timestamp and UUID to create a very unique number.
        # We can concatenate the timestamp and a portion of the UUID.
        # Alternatively, we could hash them together, but concatenation is often sufficient.
        
        # Convert UUID to integer.
        uuid_int = uuid.UUID(str(unique_id)).int

        # combine timestamp and uuid.
        unique_number = (timestamp_ns << 64) | uuid_int

        return unique_number
    
    @staticmethod
    def send_email():
        pass
    
    @staticmethod
    def merge_tool_message_contents(tool_msgs: List[BaseMessage]) -> dict:
        def extract_json_from_string(input_string):
            """
            Extracts a JSON dictionary from a string that may contain additional characters.
            Args:
                input_string (str): The input string to parse.
            Returns:
                dict or None: A dictionary if valid JSON is found, or None if not.
            """
            try:
                # Find the first and last curly braces
                start_index = input_string.find('{')
                end_index = input_string.rfind('}')

                if start_index == -1 or end_index == -1:
                    return None  # No curly braces found

                json_string = input_string[start_index:end_index + 1]

                # Attempt to parse the extracted string as JSON
                return json.loads(json_string)

            except json.JSONDecodeError:
                # Attempt to fix common errors and re-try.
                # Remove any non-printable or control characters.
                cleaned_string = re.sub(r'[^\x20-\x7E]+', '', json_string)

                # Attempt to fix single quotes to double quotes, if needed.
                cleaned_string = re.sub(r"'([^']+)'", r'"\1"', cleaned_string)

                try:
                    return json.loads(cleaned_string)
                except json.JSONDecodeError:
                    return None  # Could not parse as JSON

            except Exception:
                return None
        merged_dict: dict = {}
        for message in reversed(tool_msgs):
            if isinstance(message, ToolMessage):
                try:
                    content_dict = extract_json_from_string(message.content) # Safe if content is a valid dictionary string.
                    if content_dict is not None:
                        for key, value in content_dict.items():
                            if key not in merged_dict or not merged_dict[key] or merged_dict[key] == "null":
                                merged_dict[key] = value
                    else: print(message.content)
                except (SyntaxError, NameError, TypeError) as e:
                    print(f"Error parsing ToolMessage content: {e}")
                    pass
        return merged_dict
    
    @staticmethod
    def extract_tool_messages(messages):
        tool_messages = []
        for message in messages:
            if isinstance(message, ToolMessage):
                if message.name and "transfer_to_" in message.name: continue
                tool_messages.append(message)
        return Utilities.merge_tool_message_contents(tool_messages)

    @staticmethod
    def read_file_from_gcs(bucket_name: str, file_path: str) -> str | None:
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(file_path)
            file_content_bytes = blob.download_as_bytes()
            file_content = file_content_bytes.decode("utf-8")
            return file_content
        except exceptions.NotFound:
            print(f"Error: File '{file_path}' not found in bucket '{bucket_name}'.")
            return None
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return None
