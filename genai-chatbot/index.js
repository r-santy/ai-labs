const express = require('express');
const axios = require('axios');
const app = express();
const port = process.env.PORT || 8080;

app.use(express.json());

const endpoint = process.env.OPENAI_ENDPOINT; // e.g. https://genai-labs-cb-openai.openai.azure.com
const apiKey = process.env.OPENAI_API_KEY;
const deployment = "gpt-4.1-nano";
const apiVersion = "2024-12-01-preview"; // must be specified

app.post('/api/messages', async (req, res) => {
  const userMessage = req.body.text || "Hello";
  try {
    const url = `${endpoint}/openai/deployments/${deployment}/chat/completions?api-version=${apiVersion}`;
    const response = await axios.post(
      url,
      {
        messages: [{ role: "user", content: userMessage }]
      },
      {
        headers: {
          "api-key": apiKey,
          "Content-Type": "application/json"
        }
      }
    );
    const botReply = response.data.choices[0].message.content;
    res.json({ reply: botReply });
  } catch (error) {
    console.error("OpenAI error:", error.response?.data || error.message || error);
    res.status(500).json({ reply: "Sorry, I'm having trouble right now." });
  }
});

app.listen(port, () => {
  console.log(`Bot server listening at http://localhost:${port}`);
});
