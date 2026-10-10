from transformers import AutoTokenizer, AutoModelForCausalLM
from rag_tool import search_policy

MODEL_PATH = r"C:\Users\28101\Desktop\SCSE_26_GEX3\models\qwen3-0.6b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

SYSTEM_PROMPT = """You are a university IT support assistant. You have access to a knowledge base of university IT policies.

RULES:
1. ONLY answer based on the content returned by the search tool.
2. If the search returns "No relevant information found in the knowledge base", you MUST respond that you do not have that information and cannot provide it.
3. NEVER invent, guess, or make up a phone number, email address, or any contact detail.
4. If asked for the IT Service Desk phone number and the knowledge base does not contain it, respond exactly: "I'm sorry, I don't have the IT Service Desk phone number. Please check the university website or your student portal."
"""


def generate(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response


def run_agent(user_query: str) -> str:
    
    context = search_policy.invoke(user_query)
    draft_prompt = f"""{SYSTEM_PROMPT}

Knowledge base result:
{context}

User question: {user_query}

Answer:"""
    draft = generate(draft_prompt)
    review_prompt = f"""You are a reviewer. Check the following draft answer.

Question: {user_query}
Draft answer: {draft}

Does the draft answer only use information from the knowledge base, and does it avoid inventing any phone number or contact detail?
Reply with exactly one word: PASS or REVISE."""
    review = generate(review_prompt)

    if "REVISE" in review.upper():
        revise_prompt = f"""You are a reviser. Improve the draft answer based on the review.

Question: {user_query}
Draft answer: {draft}
Review comment: {review}

RULES:
1. ONLY use information from the knowledge base.
2. NEVER invent a phone number, email address, or any contact detail.
3. If the knowledge base does not contain the answer, say you do not have that information.

Answer:"""
        final = generate(revise_prompt)
        return final

    return draft


if __name__ == "__main__":
    while True:
        query = input("User: ")
        if query.lower() in ("exit", "quit"):
            break
        print("Assistant:", run_agent(query))