from flask import Flask, jsonify, request
import json
from flask_cors import CORS
from langchain.chat_models import ChatOllama
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts import PromptTemplate


app = Flask(__name__)


CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/', methods=['GET'])
def index():

    examples = [
    {
        "question" : "Human: What tech skills are going to be needed when doing building a liabrary web application? Just give me 4 tech names and I don't need any descriptions",
        "answer": """
        {{
            "techs": {{
                "tech1": "HTML5",
                "tech2: "React.JS",
                "tech3": "CSS",
                "tech4": "JavaScript"
         }}
        }}
        """,
    },
    {
        "question" : "Human: What tech skills are going to be needed when doing building a reservation web application? Just give me 4 tech names and I don't need any descriptions",
        "answer": """
        {{
            "techs": {{
                "tech1": "Nuxt.JS",
                "tech2: "HTML",
                "tech3": "CSS",
                "tech4": "JavaScript"
         }}
        }}
        """,
    },
    {
        "question" : "Human: What tech skills are going to be needed when doing building a SNS web application? Just give me 4 tech names and I don't need any descriptions",
        "answer": """
        {{
            "techs": {{
                "tech1": "Next.JS",
                "tech2: "HTML",
                "tech3": "CSS",
                "tech4": "JavaScript"
         }}
        }}
        """,
    },
    ]



    chat = ChatOllama(model="mistral:latest", streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

    project = request.args.get('project')

    example_prompt = PromptTemplate.from_template("System: You are a JSON data maker. Do not return any data except JSON \nHuman: {question}\nAI: {answer}")

    print(example_prompt)


    prompt = FewShotPromptTemplate(
        example_prompt = example_prompt,
        examples = examples,
        suffix= f'Human: What tech skills are going to be needed when doing {project}? Just give me 5 tech skills and I do not need any descriptions or note',
        input_variables=["project"]
    )

    print(prompt)

    chain = prompt | chat

    result = chain.invoke({
      "project": {project}
    })

    return json.dumps(result.content)

if __name__ == '__main__':
    app.run(debug=True, port=5000)