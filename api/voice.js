// Vercel Serverless Function
// This file handles all OpenAI API calls securely using environment variables

import fetch from 'node-fetch';
import FormData from 'form-data';

export default async function handler(req, res) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { audio, mimeType } = req.body;

    if (!audio) {
      return res.status(400).json({ error: 'No audio data provided' });
    }

    // Get API key from environment variable (securely stored in Vercel)
    const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

    if (!OPENAI_API_KEY) {
      return res.status(500).json({ error: 'OpenAI API key not configured' });
    }

    // Step 1: Convert speech to text using Whisper
    const transcript = await speechToText(audio, mimeType, OPENAI_API_KEY);

    // Step 2: Get AI response using GPT-4
    const aiResponse = await getAIResponse(transcript, OPENAI_API_KEY);

    // Step 3: Convert text to speech
    const audioResponse = await textToSpeech(aiResponse, OPENAI_API_KEY);

    // Return all data to frontend
    res.status(200).json({
      transcript,
      aiResponse,
      audioResponse
    });

  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: error.message || 'Internal server error' });
  }
}

async function speechToText(base64Audio, mimeType, apiKey) {
  // Convert base64 to buffer
  const audioBuffer = Buffer.from(base64Audio, 'base64');

  // Create form data
  const formData = new FormData();
  formData.append('file', audioBuffer, {
    filename: 'audio.webm',
    contentType: mimeType || 'audio/webm'
  });
  formData.append('model', 'whisper-1');

  const response = await fetch('https://api.openai.com/v1/audio/transcriptions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      ...formData.getHeaders()
    },
    body: formData
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Whisper API error: ${error}`);
  }

  const data = await response.json();
  return data.text;
}

async function getAIResponse(userMessage, apiKey) {
  const systemPrompt = `You are a knowledgeable and enthusiastic Brooklyn travel guide. You provide recommendations for:
- Restaurants and cafes
- Events and activities  
- Neighborhoods to explore
- Shopping destinations
- Attractions and landmarks

When answering questions, be specific, enthusiastic, and helpful. Mention actual places when possible.
Keep responses conversational and concise (2-4 sentences) since they'll be spoken aloud.

You specialize in Brooklyn, New York and have extensive knowledge of:
- Popular neighborhoods like Williamsburg, DUMBO, Park Slope, Brooklyn Heights
- Local restaurants, cafes, and food scenes
- Events at venues like Brooklyn Bridge Park, Prospect Park, and local galleries
- Shopping areas and boutiques
- Cultural attractions and hidden gems

Provide personalized, friendly recommendations that make visitors excited to explore Brooklyn.`;

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model: 'gpt-4',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userMessage }
      ],
      temperature: 0.7,
      max_tokens: 200
    })
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`GPT-4 API error: ${error}`);
  }

  const data = await response.json();
  return data.choices[0].message.content;
}

async function textToSpeech(text, apiKey) {
  const response = await fetch('https://api.openai.com/v1/audio/speech', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model: 'tts-1',
      input: text,
      voice: 'alloy'
    })
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`TTS API error: ${error}`);
  }

  const audioBuffer = await response.arrayBuffer();
  return Buffer.from(audioBuffer).toString('base64');
}
