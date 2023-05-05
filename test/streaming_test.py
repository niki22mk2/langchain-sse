import httpx, json
import asyncio

# Replace this with your API server URL
API_SERVER_URL = "http://localhost:8000"

async def main():
    async with httpx.AsyncClient() as client:
        message = "Hello, Chatsonic!"

        # Send the message to the API server
        async with client.stream("POST", f"{API_SERVER_URL}/chat", json={"message": message}) as response:

            # Check if the request was successful
            if response.status_code != httpx.codes.OK:
                print("Error:", response.status_code)
                return

            # Receive and print the event stream from the API server
            async for line in response.aiter_lines():
                if "[DONE]" in line:
                    break

                print(line)

if __name__ == "__main__":
    asyncio.run(main())