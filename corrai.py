import streamlit as st
import openai
import speech_recognition as sr
import time
import tempfile
from playsound import playsound
from openai import OpenAI
from streamlit_chat import message
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI


api_key = st.secrets["API_KEY"]
# Initialize OpenAI client
client = OpenAI(api_key=api_key)
openai.api_key = api_key

for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))
    st.write("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))

# Initialize the language model (GPT-3.5 Turbo)
llm = ChatOpenAI(
    openai_api_key=openai.api_key,
    temperature=1,
    model_name='gpt-3.5-turbo'
)

# Define the conversation template
template = """You are a chatbot that is friendly and has a great sense of humor.
Don't give long responses and always feel free to ask interesting questions that keep someone engaged.
You should also be a bit entertaining and not boring to talk to. Use informal language
and be curious.

Previous conversation:
{chat_history}

New human question: {question}
Response:"""

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=template
)

# Initialize conversation history in session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a chatbot that is friendly and has a great sense of humor. Don't give long responses and always feel free to ask interesting questions that keep someone engaged. You should also be a bit entertaining and not boring to talk to. Use informal language and be curious."}
    ]
    st.session_state.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# Create memory for conversation history
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=st.session_state.memory
)

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Function to load and split documents
def load_and_split_documents(file):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
    raw_documents = PyPDFLoader(file).load()
    return text_splitter.split_documents(raw_documents)

# Function to create the FAISS database
def create_faiss_db(documents):
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    return FAISS.from_documents(documents, embeddings)

# Function to search PDF content
def search_pdf_content(query):
    if 'vector_db' not in st.session_state:
        raise ValueError("Vector database not found in session state.")

    # Use the existing vector database  if 'vector_db' in st.session_state:
        db = st.session_state.vector_db

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm, 
            retriever=db.as_retriever(), 
            memory=st.session_state.memory, 
            verbose=True
        )

    # Retrieve the relevant information
    combined_info = qa_chain.run({
        'question': query, 
        'chat_history': st.session_state.memory.buffer
    })
    
    return combined_info

# Function to listen to audio input and convert to text
def listen():
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(audio.get_wav_data())
            temp_audio_file_path = temp_audio_file.name

        with open(temp_audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                prompt="Please accurately transcribe the following conversation. Include any names or technical terms that may appear."
            )

        if isinstance(transcription, str):
            return transcription
        elif isinstance(transcription, dict) and 'text' in transcription:
            return transcription['text']
        else:
            raise ValueError("Unexpected response format from transcription API")

    except Exception as e:
        st.write(f"Error in transcription: {e}")
        return None

# Function to generate a response based on the prompt
def generate_response(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})

    with st.spinner('Generating response...'):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=st.session_state['messages']
        )
        response = completion.choices[0].message['content']

    st.session_state['messages'].append({"role": "assistant", "content": response})
    return response

def respond(model_response):
    try:
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice="alloy",
            input=model_response
        )

        audio_content = response.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(audio_content)
            temp_file_path = temp_file.name

        if temp_file_path:
            playsound(temp_file_path)
        else:
            st.write("Failed to generate audio content.")

    except RuntimeError as e:
        st.write(f"Speech synthesis error: {e}")

def conversation():
    """Manages the continuous flow of the conversation."""
    st.write("Conversation started. Speak to the assistant.")
    
    while True:
        user_input = listen()
        st.write(f"User said: {user_input}")
        if user_input:
            message(user_input, is_user=True)

            # Direct all user input to search PDF content
            pdf_response = search_pdf_content(user_input)
            message(pdf_response)
            respond(pdf_response)
            time.sleep(1)

# Streamlit UI
st.title("ChatGPT Voice Assistant")
st.write("Speak to the assistant and it will respond with a voice!")

st.write(sr.Microphone.list_microphone_names())

# File uploader for PDF
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

# Load and process the PDF file if uploaded
if uploaded_file is not None:
    with st.spinner('Processing PDF...'):
        # Save the uploaded file
        file_path = "uploaded_document.pdf"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        # Load and split documents
        documents = load_and_split_documents(file_path)
        # Create the vector database and store it in the session state
        st.session_state.vector_db = create_faiss_db(documents)
    st.write("PDF file loaded, processed, and vector database created.")

# Start the conversation
if st.button("Start Conversation"):
    if 'vector_db' in st.session_state:
        conversation()
    else:
        st.write("Please upload a PDF file to start the conversation.")
