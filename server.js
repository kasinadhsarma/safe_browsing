const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const path = require('path');

const app = express();
const upload = multer();

// Python backend URL
const PYTHON_BACKEND = 'http://localhost:8000';

// Enhanced CORS configuration
app.use(cors({
    origin: '*',
    methods: ['GET', 'POST'],
    allowedHeaders: ['Content-Type', 'Authorization', 'User-Agent'],
    credentials: true
}));

app.use(express.json());

// Proxy endpoint for URL checking
app.post('/api/check-url', upload.none(), async (req, res) => {
    try {
        const { url } = req.body;
        const userAgent = req.headers['user-agent'];
        
        // Forward request to Python backend
        const response = await axios.post(`${PYTHON_BACKEND}/api/check-url`, 
            { url }, 
            { headers: { 'User-Agent': userAgent } }
        );
        
        res.json(response.data);
    } catch (error) {
        console.error('Error checking URL:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Proxy endpoint for activity logging
app.post('/api/activity', upload.none(), async (req, res) => {
    try {
        const formData = {
            ...req.body,
            browser: req.headers['user-agent']
        };

        // Forward request to Python backend
        const response = await axios.post(`${PYTHON_BACKEND}/api/activity`, formData);
        res.json(response.data);
    } catch (error) {
        console.error('Error logging activity:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Proxy endpoint for stats
app.get('/api/stats', async (req, res) => {
    try {
        const response = await axios.get(`${PYTHON_BACKEND}/api/stats`);
        res.json(response.data);
    } catch (error) {
        console.error('Error fetching stats:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Proxy endpoint for activities
app.get('/api/activities', async (req, res) => {
    try {
        const { limit = 100, offset = 0 } = req.query;
        const response = await axios.get(`${PYTHON_BACKEND}/api/activities`, {
            params: { limit, offset }
        });
        res.json(response.data);
    } catch (error) {
        console.error('Error fetching activities:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Proxy endpoint for alerts
app.get('/api/alerts', async (req, res) => {
    try {
        const response = await axios.get(`${PYTHON_BACKEND}/api/alerts`);
        res.json(response.data);
    } catch (error) {
        console.error('Error fetching alerts:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Error handling middleware
