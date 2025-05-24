# debug_gemini.py - Debug Gemini API issues
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

print("🔍 Debugging Gemini API Connection\n")

# 1. Check API key
api_key = os.getenv("GEMINI_API_KEY")
print(f"1️⃣ API Key Status:")
if api_key:
    print(f"   ✓ Found (starts with): {api_key[:10]}...")
    print(f"   Length: {len(api_key)} characters")
else:
    print("   ❌ No API key found!")
    print("   Please add GEMINI_API_KEY to your .env file")
    exit(1)

# 2. Configure Gemini
print("\n2️⃣ Configuring Gemini...")
try:
    genai.configure(api_key=api_key)
    print("   ✓ Configuration successful")
except Exception as e:
    print(f"   ❌ Configuration failed: {e}")
    exit(1)

# 3. List available models
print("\n3️⃣ Available Models:")
try:
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"   - {model.name}")
except Exception as e:
    print(f"   ❌ Failed to list models: {e}")
    print("   This might indicate an API key issue")

# 4. Test simple generation
print("\n4️⃣ Testing Simple Generation:")
try:
    model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
    response = model.generate_content("Say hello in one word")
    print(f"   ✓ Response: {response.text}")
except Exception as e:
    print(f"   ❌ Generation failed: {e}")
    print(f"   Error type: {type(e).__name__}")
    
    # Parse specific errors
    error_str = str(e)
    if "API_KEY_INVALID" in error_str:
        print("\n   🔧 Fix: Your API key is invalid")
        print("   1. Go to https://makersuite.google.com/app/apikey")
        print("   2. Create a new API key")
        print("   3. Update your .env file")
    elif "RATE_LIMIT_EXCEEDED" in error_str:
        print("\n   🔧 Fix: Rate limit exceeded")
        print("   1. Wait a few minutes")
        print("   2. Or create a new API key")
    elif "User location is not supported" in error_str:
        print("\n   🔧 Fix: Gemini not available in your region")
        print("   Consider using a VPN or different API")

# 5. Test embedding generation
print("\n5️⃣ Testing Embedding Generation:")
try:
    result = genai.embed_content(
        model="models/embedding-001",
        content="Test embedding",
        task_type="retrieval_document"
    )
    print(f"   ✓ Embedding dimension: {len(result['embedding'])}")
except Exception as e:
    print(f"   ❌ Embedding failed: {e}")

# 6. Test with different models
print("\n6️⃣ Testing Alternative Models:")
models_to_test = ['gemini-pro', 'gemini-1.5-flash', 'gemini-pro-vision']
for model_name in models_to_test:
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Hi")
        print(f"   ✓ {model_name}: Works")
    except Exception as e:
        print(f"   ❌ {model_name}: {type(e).__name__}")

# 7. Check API key format
print("\n7️⃣ API Key Validation:")
if api_key:
    if api_key.startswith("AIza"):
        print("   ✓ Key format looks correct (starts with AIza)")
    else:
        print("   ⚠️  Key doesn't start with expected prefix")
    
    if " " in api_key or "\n" in api_key:
        print("   ❌ Key contains whitespace - please check .env file")
    
    if len(api_key) < 30:
        print("   ❌ Key seems too short")

print("\n📝 Summary:")
print("If all tests fail, please:")
print("1. Verify your API key at https://makersuite.google.com/app/apikey")
print("2. Make sure the key is enabled")
print("3. Check if you have any IP restrictions")
print("4. Try creating a new API key")