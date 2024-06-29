from openai import OpenAI

client = OpenAI(organization='',
                api_key='')

def generate(user_prompt,
             system_prompt, 
             model='gpt-4o', 
             temperature=0, 
             max_tokens=250, 
             frequency_penalty=0,
             presence_penalty=0
             ):
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        messages=[
            {"role": "assistant", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content