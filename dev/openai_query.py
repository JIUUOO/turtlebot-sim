import json
from pathlib import Path
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv(Path(__file__).with_name(".env"), override=True)
# .env > OPENAI_API_KEY="YOUR KEY"

client = OpenAI()

# 1. LLM이 채워주어야 할 데이터의 '규격(설명서)' 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "control_robot",
            "description": "로봇의 선속도와 각속도를 제어합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "linear_velocity": {"type": "number", "description": "앞/뒤 속도(m/s)"},
                    "angular_velocity": {"type": "number", "description": "회전 속도(rad/s)"}
                },
                "required": ["linear_velocity", "angular_velocity"]
            }
        }
    }
]

# 2. 질문을 던지고 규격에 맞는 답변(JSON) 받기
user_command = "시계방향으로 90도 회전해줘"
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": user_command}],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "control_robot"}}
)

# 3. 결과 해석하기
tool_call = response.choices[0].message.tool_calls[0]
arguments = json.loads(tool_call.function.arguments)
print(f"해석된 결과: {arguments}")
