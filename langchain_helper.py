from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()


def generate_pet_name(animal_type: str, pet_color: str):
    llm = OpenAI(temperature=0.5)
    prompt_template_name = PromptTemplate(
        input_variables=["animal_type", "pet_color"],
        template="I have a {animal_type} pet and I want a cool name for it. It is {pet_color} in color. Suggest"
                 "me five cool name for my pet"
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="pet_name")
    response = name_chain({'animal_type': animal_type, 'pet_color': pet_color})
    return response


if __name__ == "__main__":
    print(generate_pet_name("dog","white"))
