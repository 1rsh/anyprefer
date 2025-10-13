from utils.configs import OPENAI_API_KEY, OPENAI_BASE_URL

OPENAI = "OPENAI"

GPT_4O = "gpt-4o"
GPT_4_1 = "gpt-4.1"
GPT_5 = "gpt-5"
GPT_5_MINI = "gpt-5-mini"
GPT_5_NANO = "gpt-5-nano"
GPT_4O_MINI = "gpt-4o-mini"

PROVIDER_INFORMATION = {
    OPENAI: {
        "API": (OPENAI_API_KEY, OPENAI_BASE_URL),
        "MODEL_ID": {
            GPT_4O: "gpt-4o",
            GPT_4_1: "gpt-4.1",
            GPT_5: "gpt-5",
            GPT_5_MINI: "gpt-5-mini",
            GPT_5_NANO: "gpt-5-nano",
            GPT_4O_MINI: "gpt-4o-mini",
        }
    }
}