import os
import dspy

local_uri = os.getenv("LOCAL_LLM_URI", "ollama://phi3:mini")
print("Local LLM URI:", local_uri)

try:
    dspy.configure(lm=dspy.LM(local_uri))
    print("✅ DSPy successfully connected to local model!")
except Exception as e:
    print("❌ Error connecting to model:", e)
