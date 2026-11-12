from app.services.llm_service import generate_response


SYSTEM_PROMPT = """
You are a helpful assistant.
Return only JSON.
"""


USER_PROMPT = """
Analyze this statement:

"The biggest mistake I made as a founder
was hiring people too quickly."

Return:

{
    "interesting": true,
    "reason": "..."
}
"""


response = generate_response(
    SYSTEM_PROMPT,
    USER_PROMPT
)

print(response)