const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');

const PORT = Number(process.env.PORT) || 3001;
const uploadDir = path.join(__dirname, 'uploads');
const videoExtensions = new Set(['.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm']);

fs.mkdirSync(uploadDir, { recursive: true });

const storage = multer.diskStorage({
  destination: (_req, _file, callback) => {
    callback(null, uploadDir);
  },
  filename: (_req, file, callback) => {
    const extension = path.extname(file.originalname).toLowerCase();
    const safeExtension = videoExtensions.has(extension) ? extension : '.mp4';
    callback(null, `${Date.now()}-${Math.random().toString(16).slice(2)}${safeExtension}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 500 * 1024 * 1024 },
  fileFilter: (_req, file, callback) => {
    const extension = path.extname(file.originalname).toLowerCase();
    const isVideo =
      file.mimetype.startsWith('video/') ||
      file.mimetype === 'application/octet-stream' ||
      videoExtensions.has(extension);
    if (!isVideo) {
      callback(new Error('Only video files are accepted'));
      return;
    }
    callback(null, true);
  },
});

const app = express();

app.get('/health', (_req, res) => {
  res.json({ ok: true });
});

app.post('/upload', upload.single('video'), (req, res) => {
  if (!req.file) {
    res.status(400).json({ error: 'Missing video file' });
    return;
  }
  console.log('Uploaded:', req.file.originalname);
  res.status(201).json({
    status: 'uploaded',
    filename: req.file.originalname,
    storedAs: req.file.filename,
    size: req.file.size,
  });
});

app.use((error, _req, res, _next) => {
  res.status(400).json({ error: error.message || 'Upload failed' });
});

app.listen(PORT, () => {
  console.log(`Upload server listening on http://localhost:${PORT}`);
});
