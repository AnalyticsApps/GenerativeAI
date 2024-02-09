from openai import OpenAI
import streamlit as st
import subprocess

st.title("Secure Python Code Generator")

def run_shell_script(input_text):
    script_path = "code_scan_script.sh"
    result = subprocess.run(["bash", script_path, input_text], capture_output=True, text=True)
    return result.stdout.strip()


# Sidebar for config
st.sidebar.header("OpenAI Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
selModel = st.sidebar.selectbox(label="Select a Model", options=["gpt-3.5-turbo", "gpt-4"], index = 0)

GEN_CODE = """
Context:
You are a proficient Python programmer tasked with providing correct and executable Python code to users.
Your goal is to generate code snippets based on user input. Follow the specified guidelines below:

Guidelines:
    1) Code Format:
       Wrap the generated Python code within python code markdown.
    2) Dependency Mention:
       If the code requires any external package dependency, mention it using the following syntax:
       To install the <dependency> package, run: python -m pip install dependency1 dependency2
    3) Code Completeness:
       Do not abbreviate or omit any part of the code. Provide a complete and executable Python code snippet.
    4) Single Code Generation:
       Generate a single Python code snippet, not multiple.

Example:
    User Input:
        Consider a scenario where the user wants to calculate sum of 2 numbers.

    Generated Response:
        ```
            n = int(input("Enter a number: "))
            factorial = 1

            # Calculate factorial
            for i in range(1, n + 1):
                factorial *= i

            print(f"The factorial of {n} is {factorial}")

        ```
    Dependency Mention:
        To install the required package for this code, run: ```pip install math ```

Now to get started, briefly introduce yourself 2-3 sentences.
Then provide 1 example user input using bullet points and response generated.

"""

if "script_called" not in st.session_state:
    st.session_state.script_called = False


# Check if API key is provided
if not api_key:
    st.error("Please enter your OpenAI API Key in the sidebar.")

else:
    # Set the API key if provided
    #openai.api_key = api_key
    client = OpenAI(api_key=api_key)
    if "messages" not in st.session_state :
        # system prompt includes table information, rules, and prompts the LLM to produce
        # a welcome message to the user.
        st.session_state.messages = [{"role": "system", "content": GEN_CODE}]

    # Prompt for user input and save
    # st.text_area("Provide input to generate the code.", "")
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})

    # display the existing chat messages
    for message in st.session_state.messages:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "results" in message:
                st.dataframe(message["results"])

    # If last message is not from assistant, we need to generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            response = ""
            resp_container = st.empty()
            for delta in client.chat.completions.create(
                model=selModel,
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True,
            ):
                response += (delta.choices[0].delta.content or "")
                resp_container.markdown(response)

            if st.session_state.script_called:
                code_scan_resp = run_shell_script(response)
                st.warning(f"Code Scan response: {code_scan_resp}")


            st.session_state.script_called = True

            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
