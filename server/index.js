import express from 'express';
import multer from 'multer';
import cors from 'cors';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const upload = multer({ dest: 'uploads/' });

app.use(cors());
app.use(express.json());

// Serve uploaded files
app.use('/uploads', express.static('uploads'));

// Process images endpoint
app.post('/api/process', upload.array('images', 10), (req, res) => {
  try {
    const files = req.files;
    
    // In a real application, you would process the images here
    // For now, we'll just return mock data
    
    const results = {
      images: files.map((file, index) => ({
        id: `img-${index}`,
        url: `/uploads/${file.filename}`,
        annotations: [],
        classification: `class-${index}`
      })),
      segmentationData: {
        version: '1.0',
        timestamp: new Date().toISOString(),
        results: []
      },
      classificationData: {
        version: '1.0',
        timestamp: new Date().toISOString(),
        results: []
      }
    };

    res.json(results);
  } catch (error) {
    console.error('Error processing images:', error);
    res.status(500).json({ error: 'Failed to process images' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});