import base64
import json
import logging
import requests  # Import for external API
from io import BytesIO
import tiktoken
import openai
import config

# setup openai
openai.api_key = config.openai_api_key
if config.openai_api_base is not None:
    openai.api_base = config.openai_api_base
logger = logging.getLogger(__name__)


OPENAI_COMPLETION_OPTIONS = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "request_timeout": 60.0,
}


class ChatGPT:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.logger = logging.getLogger(__name__)

    async def send_message(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        # Step 1: Try external API
        try:
            external_api_response = self.call_external_api(message)
            if external_api_response:
                return external_api_response
        except Exception as e:
            self.logger.error(f"External API failed: {e}")
        
        # Step 2: Fallback to OpenAI API if external API fails
        return await self.call_openai_api(message, dialog_messages, chat_mode)
    
    def call_external_api(self, message):
        """This function calls your external API (such as your custom API)"""
        try:
            url = "https://ar-api-08uk.onrender.com/chat/v1"
            headers = {"Content-Type": "application/json"}
            data = {"userMessage": message}
            response = requests.post(url, headers=headers, json=data)
            
            # Check if external API was successful
            if response.status_code == 200:
                api_response = response.json()
                return api_response.get("response", "")  # Or handle as per the API response
            else:
                raise Exception(f"External API returned error: {response.status_code}")
        
        except Exception as e:
            raise Exception(f"Failed to call external API: {e}")
    
    async def call_openai_api(self, message, dialog_messages, chat_mode):
        """This function calls the OpenAI API if the external API fails."""
        try:
            if self.model in {"gpt-3.5-turbo-16k", "gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4-1106-preview"}:
                messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                r = await openai.ChatCompletion.acreate(
                    model=self.model,
                    messages=messages,
                    **OPENAI_COMPLETION_OPTIONS
                )
                answer = r.choices[0].message["content"]
            else:
                raise ValueError(f"Unknown model: {self.model}")

            return answer  # Return OpenAI's response
        except Exception as e:
            self.logger.error(f"Failed to call OpenAI API: {e}")
            return "Sorry, something went wrong."
    # Other methods remain the same...

    def _generate_prompt(self, message, dialog_messages, chat_mode):
        prompt = config.chat_modes[chat_mode]["prompt_start"]
        prompt += "\n\n"

        # add chat context
        if len(dialog_messages) > 0:
            prompt += "Chat:\n"
            for dialog_message in dialog_messages:
                prompt += f"User: {dialog_message['user']}\n"
                prompt += f"Assistant: {dialog_message['bot']}\n"

        # current message
        prompt += f"User: {message}\n"
        prompt += "Assistant: "

        return prompt

    def _encode_image(self, image_buffer: BytesIO) -> bytes:
        return base64.b64encode(image_buffer.read()).decode("utf-8")

    def _generate_prompt_messages(self, message, dialog_messages, chat_mode, image_buffer: BytesIO = None):
        # Start with the system prompt based on the selected chat mode.
        prompt = config.chat_modes[chat_mode]["prompt_start"]

        # Initialize the list of messages.
        messages = [{"role": "system", "content": prompt}]
        
        # Add all previous dialog messages (user and bot interactions).
        for dialog_message in dialog_messages:
            messages.append({"role": "user", "content": dialog_message["user"]})
            messages.append({"role": "assistant", "content": dialog_message["bot"]})
                    
        # Check if an image was provided.
        if image_buffer is not None:
            # If there's an image, append the message content including the text and image.
            messages.append(
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": message,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self._encode_image(image_buffer)}",
                                "detail": "high"  # or any other detail level you need
                            }
                        }
                    ]
                }
            )
        else:
            # If there's no image, just add the text message as usual.
            messages.append({"role": "user", "content": message})

        return messages

    def _postprocess_answer(self, answer):
        answer = answer.strip()
        return answer

    def _count_tokens_from_messages(self, messages, answer, model="gpt-3.5-turbo"):
        encoding = tiktoken.encoding_for_model(model)

        if model == "gpt-3.5-turbo-16k":
            tokens_per_message = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model == "gpt-3.5-turbo":
            tokens_per_message = 4
            tokens_per_name = -1
        elif model == "gpt-4":
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-4-1106-preview":
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-4-vision-preview":
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-4o":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise ValueError(f"Unknown model: {model}")

        # input
        n_input_tokens = 0
        for message in messages:
            n_input_tokens += tokens_per_message
            if isinstance(message["content"], list):
                for sub_message in message["content"]:
                    if "type" in sub_message:
                        if sub_message["type"] == "text":
                            n_input_tokens += len(encoding.encode(sub_message["text"]))
                        elif sub_message["type"] == "image_url":
                            pass
            else:
                if "type" in message:
                    if message["type"] == "text":
                        n_input_tokens += len(encoding.encode(message["text"]))
                    elif message["type"] == "image_url":
                        pass


        n_input_tokens += 2

        # output
        n_output_tokens = 1 + len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens

    def _count_tokens_from_prompt(self, prompt, answer, model="text-davinci-003"):
        encoding = tiktoken.encoding_for_model(model)

        n_input_tokens = len(encoding.encode(prompt)) + 1
        n_output_tokens = len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens


async def transcribe_audio(audio_file) -> str:
    r = await openai.Audio.atranscribe("whisper-1", audio_file)
    return r["text"] or ""


async def generate_images(prompt, n_images=4, size="512x512"):
    r = await openai.Image.acreate(prompt=prompt, n=n_images, size=size)
    image_urls = [item.url for item in r.data]
    return image_urls


async def is_content_acceptable(prompt):
    r = await openai.Moderation.acreate(input=prompt)
    return not all(r.results[0].categories.values())
