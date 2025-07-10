import express from 'express';
import cors from 'cors';
import http from 'http';
import 'dotenv/config';
import connectDB from './config/mongodb.js'; // Assuming you have a connectDB function
import userRouter from './routes/userRouter.js';

const app = express();
app.use(cors());
app.use(express.json()); // Parse JSON request bodies

const server = http.createServer(app); // Create HTTP server instance

// Sample Route
app.get('/', (req, res) => {
    res.send('API is running...');
});

async function startServer() {
    try {
        await connectDB(); // Connect to MongoDB
        const PORT = process.env.PORT || 8000;

        server.listen(PORT, () => {
            console.log(`🚀 Server running on port ${PORT}`);
        });

    } catch (error) {
        console.error("❌ Failed to connect:", error.message);
        process.exit(1);
    }
}

app.use('/api/user', userRouter);
startServer();
