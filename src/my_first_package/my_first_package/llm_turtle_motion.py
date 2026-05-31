#!/usr/bin/env python3
import os
import sys
import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from openai import OpenAI


class LlmTurtleController(Node):
    def __init__(self):
        # ROS 2 노드 초기화 (노드 이름: llm_turtle_controller)
        super().__init__('llm_turtle_controller')

        # 터틀심 속도 제어 토픽(/cmd_vel)을 발행할 Publisher 생성
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.client = OpenAI()

        self.get_logger().info('=== LLM 터틀심 제어 에이전트 노드가 시작되었습니다 ===')
        self.get_logger().info('명령 예시: "앞으로 2미터 전진해줘", "시계방향으로 90도 회전해"')

    def ask_llm_and_move(self, user_command):
        """
        사용자의 자연어 명령을 받아 OpenAI API로 분석한 뒤, 터틀심을 구동하는 핵심 메서드
        """
        self.get_logger().info(f"사용자 명령 분석 중: '{user_command}'")

        # LLM에게 '추출해야 할 함수의 형태(스펙)'를 정의해 주는 도구(Tools) 리스트
        # 이를 통해 LLM은 텍스트 대답 대신 정밀한 구조적 데이터(JSON)를 리턴
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "control_robot",
                    "description": "로봇의 선속도(전진/후진)와 각속도(회전)를 제어합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "linear_velocity": {
                                "type": "number",
                                "description": "앞으로 가려면 양수(m/s), 뒤로 가려면 음수(m/s). 정지 시 0"
                            },
                            "angular_velocity": {
                                "type": "number",
                                "description": "반시계방향(좌회전)은 양수(rad/s), 시계방향(우회전)은 음수(rad/s). 정지 시 0"
                            }
                        },
                        "required": ["linear_velocity", "angular_velocity"]
                    }
                }
            }
        ]

        try:
            # OpenAI Chat Completion API 호출
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "너는 인간의 명령을 로봇 제어 파라미터로 변환하는 AI 로봇 인터페이스이다."},
                    {"role": "user", "content": user_command}
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "control_robot"}} # 이 함수를 반드시 쓰도록 강제
            )

            # LLM이 판단해서 채워준 함수의 인자(Arguments)값 추출
            tool_call = response.choices[0].message.tool_calls[0]
            arguments = json.loads(tool_call.function.arguments)

            linear_v = arguments.get("linear_velocity", 0.0)
            angular_v = arguments.get("angular_velocity", 0.0)

            self.get_logger().info(f"▶ LLM 해석 결과 -> 선속도: {linear_v} m/s, 각속도: {angular_v} rad/s")

            # ROS 2 메시지 생성 및 발행
            twist_msg = Twist()
            twist_msg.linear.x = float(linear_v)   # 전진/후진 속도 지정
            twist_msg.angular.z = float(angular_v) # 회전 속도 지정

            # 터틀심에게 토픽 전송 (거북이가 움직이는 시점)
            self.cmd_vel_pub.publish(twist_msg)
            self.get_logger().info("✔ ROS 2 토픽 발행 완료!")

        except Exception as e:
            self.get_logger().error(f"오류가 발생했습니다: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = LlmTurtleController()

    try:
        while rclpy.ok():
            user_input = input("\n[명령을 입력하세요 (종료하려면 q)]: ")
            if user_input.lower() == 'q':
                break
            if user_input.strip() == '':
                continue

            # 자연어 명령을 해석하고 로봇을 움직이는 함수 호출
            node.ask_llm_and_move(user_input)

            # ROS 2 이벤트 통신 처리 (스핀을 짧게 주어 이벤트 처리 유도)
            rclpy.spin_once(node, timeout_sec=0.1)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
