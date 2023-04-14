import openai
import os
from dotenv import load_dotenv

load_dotenv()

# set up the OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# set up the GPT-3 generator
model_engine = "text-davinci-003"
max_tokens = 50
temperature = 0.5

if __name__ == '__main__':
    while True:
        try:
            # prompt for the generator to use
            prompt = str(input("Write your prompt: "))
            if prompt.lower() == "thanks":
                break

            # generate the text
            response = openai.Completion.create(
                engine=model_engine,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # print the generated text
            print(response.choices[0].text)
        except ValueError:
            print('You entered unexpected value')
