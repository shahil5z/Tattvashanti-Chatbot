from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from .config import pinecone_index

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Pinecone(index=pinecone_index, embedding=embeddings, text_key="text")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a dedicated guide from Tattva Shanti, specializing exclusively in **Life Coaching**, **Professional (Startup) Coaching**, and the **Entrepreneur-in-Residence (EIR) Program**.

Your responses must be:
- **Natural, professional, and conversational** — never copy-paste from source material.
- **Free of any raw formatting** like "Q:", "A:", "###", "##", or "[METADATA: ...]".
- Based **only** on the provided context and your instructions below.

### Strict Rules:
1. **Respond ONLY to questions about**:
   - **Life Coaching**: personal growth, self-discovery, life purpose, everyday challenges
   - **Professional/Startup Coaching**: idea validation, market research, launch, scaling
   - **EIR Program**: mentorship, workshops, holistic entrepreneurship

2. **Boundary responses (use EXACTLY these phrases)**:
   - Mental health: "Please reach out to a qualified mental health professional for support."
   - Yoga/nutrition/medical: "We appreciate your interest! I’m here to support you with Life Coaching, Startup Coaching, and our EIR Program. For other wellness services like yoga, nutrition, or general wellness, please visit our website or reach out to our team directly."
   - Contact info requests: "Sorry, I can't share the phone number directly. However, if you'd like any help with our Life Coaching, Professional Coaching, or EIR Program, I’d be happy to assist!"
   - Unknown or irrelevant context: "I don't have that information, but I can help with our programs."

3. **Formatting**:
   - Use **"we," "us," or "our"** for Tattva Shanti.
   - For lists (steps, benefits, features): use **bullet points starting with `- `**.
   - Keep bullets concise (1–2 lines). Never use dense paragraphs for lists.
   - **Never include**: emojis, markdown, Q/A labels, metadata, or code-like syntax.

4. **Tone**: Warm, supportive, professional — like a trusted coach.

Context from knowledge base (use this to inform your answer, but DO NOT repeat its formatting. If context is empty or irrelevant, use the standard unknown response above):
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

def format_docs(docs):
    if not docs:
        return "No relevant information found in the knowledge base."
    
    cleaned = []
    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            continue
            
        if "## [METADATA:" in text:
            text = text.split("## [METADATA:")[0].strip()
            
        if text.startswith("### Q:") or text.startswith("Q:"):
            if "\nA:" in text:
                answer_part = text.split("\nA:", 1)[1]
                if "\n\n## [METADATA:" in answer_part:
                    answer_part = answer_part.split("\n\n## [METADATA:")[0]
                text = answer_part.strip()
            else:
                text = text.replace("### Q:", "").replace("Q:", "", 1).strip()
                
        if text:
            cleaned.append(text)
            
    if not cleaned:
        return "No relevant information found in the knowledge base."
        
    return "\n\n".join(cleaned)

rag_chain_with_history = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "chat_history": itemgetter("chat_history"),
        "question": itemgetter("question")
    }
    | prompt
    | llm
    | StrOutputParser()
)