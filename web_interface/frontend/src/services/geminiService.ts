/**
 * Client-side Gemini API service for GitHub Pages deployment
 */

interface GeminiResponse {
  candidates: Array<{
    content: {
      parts: Array<{
        text: string;
      }>;
    };
  }>;
}

export class GeminiService {
  private apiKey: string;
  private baseUrl = 'https://generativelanguage.googleapis.com/v1beta/models';

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  async generateText(prompt: string, model: string = 'gemini-1.5-flash'): Promise<string> {
    try {
      const response = await fetch(
        `${this.baseUrl}/${model}:generateContent?key=${this.apiKey}`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            contents: [
              {
                parts: [
                  {
                    text: prompt,
                  },
                ],
              },
            ],
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: GeminiResponse = await response.json();
      
      if (data.candidates && data.candidates[0]?.content?.parts?.[0]?.text) {
        return data.candidates[0].content.parts[0].text;
      }
      
      throw new Error('No response generated');
    } catch (error) {
      console.error('Gemini API error:', error);
      throw error;
    }
  }

  async listModels(): Promise<Array<{ id: string; name: string; provider: string; status: string }>> {
    return [
      {
        id: 'gemini-1.5-flash',
        name: 'Gemini 1.5 Flash',
        provider: 'Google',
        status: 'available',
      },
      {
        id: 'gemini-1.5-pro',
        name: 'Gemini 1.5 Pro',
        provider: 'Google',
        status: 'available',
      },
    ];
  }
}

// Create a singleton instance
let geminiService: GeminiService | null = null;

export const getGeminiService = (): GeminiService | null => {
  if (!geminiService) {
    const apiKey = process.env.REACT_APP_GEMINI_API_KEY;
    if (apiKey) {
      geminiService = new GeminiService(apiKey);
    }
  }
  return geminiService;
};

export default GeminiService;