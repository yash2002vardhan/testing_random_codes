from openai import OpenAI
import openai
from dotenv import load_dotenv
import os
from pydantic import BaseModel

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Vision API - Image Analysis
# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "What's in this image?"},
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
#                         "detail": "low",
#                     },
#                 },
#             ],
#         }
#     ],
#     max_tokens=100,
# )
# print(response.choices[0].message.content)


# Image Generation API (DALL-E)
#we also have a client.images.edit endpoint (works with Dalle 2) where you provide the base image and the image where you want to add things by masking that area
# there is a client.images.create_variations endpoint as well (dalle 2 only)
# response = client.images.generate( 
#     model = "dall-e-3",
#     prompt = "a white labrador retriever playing with an astronaut on the Saturn rings",
#     size = "1024x1024",
#     quality = "standard",
#     n = 1
# )

# print(response.data[0].url)


# Chat API with Audio Preview
# response = client.chat.completions.create(
#     model = "gpt-4o-audio-preview",
#     modalities = ['text', 'audio'],
#     audio = {"voice" : "alloy", "format" :"wav"},
#     messages = [
#         {
#             "role" : "user",
#             "content" : "is a golden retrievar a good family dog ?"
#         }
#     ]
# )

# print(response)


# Text-to-Speech API
# response = client.audio.speech.create(
#     model = "tts-1",
#     voice = "nova",
#     input = "hello, how are you?"
# )

# Speech-to-Text API (Whisper)
# response = client.audio.transcriptions.create( # we also have client.audio.translations.create
#     model = "whisper-1",
#     file = open("audio.mp3", "rb"),
#     # response_format = "text", default is json
#     # prompt = "transcribe the following audio" , helps with specific details
# )

# print(response.text)


# Embeddings API
# response = client.embeddings.create(
#     input = "hell how are you",
#     model = "text-embedding-3-small",
#     # dimensions = 1536 # default is 1536
# )

# print(response.data[0].embedding)

# Moderation API
# response = client.moderations.create(
#     model = "omni-moderation-latest",
#     input = "I want to kill myself"
# )

# print(response)

#Structured Outputs
# class User(BaseModel):
#     id: int
#     name: str
#     email: str


# response = client.beta.chat.completions.parse(
#     model = "gpt-4o-2024-08-06",
#     messages = [
#         {"role": "user", "content": "Extract the user id , name and email"},
#         {"role": "user", "content" : "I am Yashvardhan Goel, my id is 1234567890, my email is yashvardhan@gmail.com"}
#     ],
#     response_format = User
# )

# print(response.choices[0].message.parsed)


# Predict Output
code = """
class User {
  firstName: string = "";
  lastName: string = "";
  username: string = "";
}

export default User;
"""

refactor_prompt = """
Replace the "username" property with an "email" property. Respond only 
with code, and with no markdown formatting.
"""

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": refactor_prompt
        },
        {
            "role": "user",
            "content": code
        }
    ],
    prediction={
        "type": "content",
        "content": code
    },
    stream = True
)

# print(completion)
# print(completion.choices[0].message.content) # non streaming response

# for chunk in completion:
#     if chunk.choices[0].delta.content is not None:
#         print(chunk.choices[0].delta.content) # streaming response
