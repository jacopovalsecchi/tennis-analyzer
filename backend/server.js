const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');
const { promisify } = require('util');

const execFileAsync = promisify(execFile);

const PORT = Number(process.env.PORT) || 3001;
const projectRoot = path.join(__dirname, '..');
const uploadDir = path.join(__dirname, 'uploads');
const resultsDir = path.join(__dirname, 'results');
const videoExtensions = new Set(['.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm']);
const venvPython = path.join(projectRoot, '.venv', 'bin', 'python');
const python = fs.existsSync(venvPython) ? venvPython : 'python3';
const ffprobe = process.env.FFPROBE || 'ffprobe';
const ANALYSIS_TIMEOUT_MS = 15 * 60 * 1000;
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseAnonKey = process.env.SUPABASE_ANON_KEY;
const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const authRequired = process.env.SUPABASE_AUTH_REQUIRED === 'true';
const supabaseConfigured = Boolean(supabaseUrl && supabaseAnonKey);
const persistenceConfigured = Boolean(supabaseUrl && supabaseServiceRoleKey);

fs.mkdirSync(uploadDir, { recursive: true });
fs.mkdirSync(resultsDir, { recursive: true });

function formChoice(value, allowed, fallback) {
  if (typeof value !== 'string') {
    return fallback;
  }
  const normalized = value.trim().toLowerCase();
  return allowed.includes(normalized) ? normalized : fallback;
}

function readStrokes(csvPath) {
  const lines = fs.readFileSync(csvPath, 'utf8').trim().split('\n');
  const columns = lines[0].split(',');
  return lines.slice(1).filter(line => line.trim()).map(line => {
    const cells = line.split(',');
    return Object.fromEntries(columns.map((column, index) => [column, cells[index] ?? '']));
  });
}

function countStrokes(strokes) {
  const counts = { shots: strokes.length, forehands: 0, backhands: 0, oneHanded: 0, twoHanded: 0 };
  for (const stroke of strokes) {
    if (stroke.stroke_type === 'forehand') {
      counts.forehands += 1;
    } else if (stroke.stroke_type === 'backhand') {
      counts.backhands += 1;
      if (stroke.backhand_style === 'two-handed') {
        counts.twoHanded += 1;
      } else {
        counts.oneHanded += 1;
      }
    }
  }
  return counts;
}

async function getVideoDuration(videoPath) {
  const { stdout } = await execFileAsync(ffprobe, [
    '-v',
    'error',
    '-show_entries',
    'format=duration',
    '-of',
    'default=noprint_wrappers=1:nokey=1',
    videoPath,
  ]);
  const duration = Number.parseFloat(stdout.trim());
  if (!Number.isFinite(duration)) {
    throw new Error('Unable to determine video duration');
  }
  return duration;
}

function logBlock(title, text) {
  const body = (text || '').trim();
  console.log(`--- ${title} ---`);
  if (body) {
    console.log(body);
  }
}

function logResults(videoName, options, analysis) {
  console.log('--- Analysis complete ---');
  console.log(`Video: ${videoName}`);
  console.log(`Dominant arm: ${options.dominantArm}`);
  console.log(`Backhand preference: ${options.backhand || 'not set'}`);
  console.log(`Status: ${analysis.status}`);
  console.log(
    `Shots: ${analysis.counts.shots} ` +
      `(forehands ${analysis.counts.forehands}, ` +
      `backhands ${analysis.counts.backhands}, ` +
      `one-handed ${analysis.counts.oneHanded}, ` +
      `two-handed ${analysis.counts.twoHanded})`,
  );
  for (const stroke of analysis.strokes) {
    const style = stroke.backhand_style ? ` ${stroke.backhand_style}` : '';
    console.log(
      `  Shot ${stroke.shot}: ${stroke.stroke_type}${style}, ` +
        `${stroke.striking_arm} arm, ` +
        `contact ${stroke.t_contact}s, ` +
        `elbow ${stroke.elbow_angle_deg}°, ` +
        `racket-torso ${stroke.racket_torso_angle_deg}°, ` +
        `wrist speed ${stroke.wrist_speed}`,
    );
  }
  console.log(`Overlay: ${analysis.overlay}`);
  console.log(`Strokes CSV: ${analysis.strokesCsv}`);
  console.log(`Features video: ${analysis.features}`);
  for (const shot of analysis.comments) {
    console.log(`  ${shot.time}s ${shot.label}: ${shot.comment}`);
  }
}

async function runAnalysis(videoPath, options) {
  const stem = path.basename(videoPath, path.extname(videoPath));
  const outputDir = path.join(resultsDir, stem);
  fs.mkdirSync(outputDir, { recursive: true });
  const duration = await getVideoDuration(videoPath);
  const run = async (script, args) => {
    console.log(`Status: running ${script}`);
    const result = await execFileAsync(python, [path.join(projectRoot, script), ...args], {
      cwd: projectRoot,
      timeout: ANALYSIS_TIMEOUT_MS,
      maxBuffer: 10 * 1024 * 1024,
    });
    logBlock(script, `${result.stdout || ''}${result.stderr || ''}`);
    return result;
  };

  await run('try_mediapipe.py', [videoPath, '--output', outputDir]);
  const strokesCsv = path.join(outputDir, `${stem}_strokes.csv`);
  const featureArgs = [
    path.join(outputDir, `${stem}_keypoints.csv`),
    '--video',
    videoPath,
    '--output',
    outputDir,
    '--dominant-arm',
    options.dominantArm,
  ];
  if (options.backhand) {
    featureArgs.push('--backhand', options.backhand);
  }
  await run('extract_features.py', featureArgs);
  const commentsPath = path.join(outputDir, `${stem}_comments.json`);
  await run('comment_strokes.py', [
    path.join(outputDir, `${stem}_keypoints.csv`),
    strokesCsv,
    '--video',
    videoPath,
    '--output',
    commentsPath,
  ]);

  const strokes = readStrokes(strokesCsv);
  const comments = JSON.parse(fs.readFileSync(commentsPath, 'utf8')).comments ?? [];
  return {
    status: 'complete',
    duration,
    outputDir,
    overlay: path.join(outputDir, `${stem}_overlay.mp4`),
    strokesCsv,
    features: path.join(outputDir, `${stem}_features.mp4`),
    strokes,
    counts: countStrokes(strokes),
    comments,
  };
}

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

async function supabaseRest(resource, options = {}, key = supabaseServiceRoleKey) {
  if (!supabaseUrl || !key) {
    throw new Error('Supabase persistence is not configured');
  }
  const response = await fetch(`${supabaseUrl}/rest/v1/${resource}`, {
    ...options,
    headers: {
      apikey: key,
      Authorization: `Bearer ${key}`,
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
  });
  const body = await response.text();
  let parsed = null;
  if (body) {
    try {
      parsed = JSON.parse(body);
    } catch {
      parsed = body;
    }
  }
  if (!response.ok) {
    throw new Error(parsed?.message || parsed?.hint || parsed || `Supabase request failed (${response.status})`);
  }
  return parsed;
}

async function authenticate(req, res, next) {
  const authorization = req.get('authorization') || '';
  const token = authorization.startsWith('Bearer ')
    ? authorization.slice('Bearer '.length).trim()
    : null;

  if (!token) {
    if (authRequired) {
      res.status(401).json({ error: 'Authentication required' });
      return;
    }
    req.user = null;
    next();
    return;
  }

  if (!supabaseConfigured) {
    res.status(503).json({ error: 'Supabase authentication is not configured' });
    return;
  }

  try {
    const response = await fetch(`${supabaseUrl}/auth/v1/user`, {
      headers: {
        apikey: supabaseAnonKey,
        Authorization: `Bearer ${token}`,
      },
    });
    const user = response.ok ? await response.json() : null;
    if (!user?.id) {
      throw new Error('Invalid authentication token');
    }
    req.user = user;
    next();
  } catch {
    res.status(401).json({ error: 'Invalid authentication token' });
  }
}

async function getProfile(user) {
  if (!user || !persistenceConfigured) {
    return null;
  }

  const data = await supabaseRest(
    'profiles?on_conflict=id',
    {
      method: 'POST',
      headers: { Prefer: 'resolution=merge-duplicates,return=representation' },
      body: JSON.stringify({
        id: user.id,
        display_name: user.user_metadata?.full_name || user.user_metadata?.name || null,
        avatar_url: user.user_metadata?.avatar_url || user.user_metadata?.picture || null,
      }),
    },
  );
  return data?.[0] || null;
}

async function createAnalysisJob(user, values) {
  if (!user || !persistenceConfigured) {
    return null;
  }

  const data = await supabaseRest('analysis_jobs', {
    method: 'POST',
    headers: { Prefer: 'return=representation' },
    body: JSON.stringify({
      user_id: user.id,
      original_filename: values.originalFilename,
      stored_filename: values.storedFilename,
      dominant_arm: values.dominantArm,
      backhand: values.backhand,
      status: 'processing',
    }),
  });
  return data?.[0]?.id || null;
}

async function saveAnalysisResult(jobId, analysis) {
  if (!jobId || !persistenceConfigured) {
    return;
  }

  await supabaseRest('analysis_results?on_conflict=job_id', {
    method: 'POST',
    headers: { Prefer: 'resolution=merge-duplicates,return=minimal' },
    body: JSON.stringify({
      job_id: jobId,
      duration_seconds: analysis.duration,
      strokes: analysis.strokes,
      counts: analysis.counts,
      comments: analysis.comments,
      overlay_path: analysis.overlay,
      strokes_csv_path: analysis.strokesCsv,
      features_path: analysis.features,
    }),
  });
  await supabaseRest(`analysis_jobs?id=eq.${encodeURIComponent(jobId)}`, {
    method: 'PATCH',
    headers: { Prefer: 'return=minimal' },
    body: JSON.stringify({ status: 'complete', completed_at: new Date().toISOString() }),
  });
}

async function failAnalysisJob(jobId, message) {
  if (!jobId || !persistenceConfigured) {
    return;
  }
  await supabaseRest(`analysis_jobs?id=eq.${encodeURIComponent(jobId)}`, {
    method: 'PATCH',
    headers: { Prefer: 'return=minimal' },
    body: JSON.stringify({ status: 'failed', error_message: message }),
  });
}

const latestFile = path.join(resultsDir, 'latest.json');

function readLatest() {
  if (!fs.existsSync(latestFile)) {
    return null;
  }
  return JSON.parse(fs.readFileSync(latestFile, 'utf8'));
}

function saveLatest(record) {
  fs.writeFileSync(latestFile, JSON.stringify(record));
}

function commentsForClient(comments) {
  return (comments ?? []).map(shot => {
    const image = shot.image ? path.basename(shot.image) : null;
    return {
      time: shot.time,
      frame: shot.frame ?? null,
      label: shot.label,
      arm: shot.arm,
      comment: shot.comment,
      imageUrl: image ? `/results/latest/frames/${image}` : null,
    };
  });
}

app.get('/health', (_req, res) => {
  res.json({
    ok: true,
    authConfigured: supabaseConfigured,
    persistenceConfigured,
  });
});

app.use(authenticate);

app.get('/results/latest', (_req, res) => {
  const latest = readLatest();
  if (!latest) {
    res.status(404).json({ error: 'No results yet' });
    return;
  }
  res.json({
    filename: latest.filename,
    storedAs: latest.storedAs,
    uploadedAt: latest.uploadedAt,
    duration: latest.duration,
    strokes: latest.strokes,
    comments: commentsForClient(latest.comments),
  });
});

app.get('/results/latest/frames/:name', (req, res) => {
  const latest = readLatest();
  const image = path.basename(req.params.name);
  const framesDir = latest?.overlay ? path.join(path.dirname(latest.overlay), 'comment_frames') : null;
  const file = framesDir ? path.join(framesDir, image) : null;
  if (!image.endsWith('.jpg') || !file || !fs.existsSync(file)) {
    res.status(404).json({ error: 'Frame not found' });
    return;
  }
  res.sendFile(path.resolve(file));
});

app.get('/results/latest/overlay', (_req, res) => {
  const latest = readLatest();
  if (!latest?.overlay || !fs.existsSync(latest.overlay)) {
    res.status(404).json({ error: 'Overlay not found' });
    return;
  }
  res.sendFile(path.resolve(latest.overlay));
});

app.get('/jobs', async (req, res) => {
  if (!req.user || !persistenceConfigured) {
    res.status(503).json({ error: 'Authenticated persistence is not configured' });
    return;
  }

  try {
    const data = await supabaseRest(
      `analysis_jobs?user_id=eq.${encodeURIComponent(req.user.id)}`
        + '&select=id,original_filename,dominant_arm,backhand,status,error_message,created_at,completed_at'
        + '&order=created_at.desc',
    );
    res.json({ jobs: data });
  } catch (error) {
    res.status(500).json({ error: error.message });
    return;
  }
});

app.get('/jobs/:id', async (req, res) => {
  if (!req.user || !persistenceConfigured) {
    res.status(503).json({ error: 'Authenticated persistence is not configured' });
    return;
  }

  try {
    const data = await supabaseRest(
      `analysis_jobs?id=eq.${encodeURIComponent(req.params.id)}`
        + `&user_id=eq.${encodeURIComponent(req.user.id)}`
        + '&select=id,original_filename,dominant_arm,backhand,status,error_message,created_at,completed_at,analysis_results(*)',
    );
    const job = data?.[0];
    if (!job) {
      res.status(404).json({ error: 'Job not found' });
      return;
    }
    res.json({ job });
  } catch (error) {
    res.status(500).json({ error: error.message });
    return;
  }
});

app.post('/upload', upload.single('video'), async (req, res) => {
  if (!req.file) {
    res.status(400).json({ error: 'Missing video file' });
    return;
  }
  let profile;
  try {
    profile = await getProfile(req.user);
  } catch (error) {
    res.status(500).json({ error: error.message });
    return;
  }
  const dominantArm = formChoice(
    req.body.dominantArm,
    ['left', 'right'],
    profile?.handedness || 'right',
  );
  const backhand = formChoice(
    req.body.backhand,
    ['one-handed', 'two-handed'],
    profile?.backhand_style || null,
  );
  let jobId;
  try {
    console.log('--- Upload ---');
    console.log(`Video: ${req.file.originalname}`);
    console.log(`Stored as: ${req.file.path}`);
    console.log(`Dominant arm: ${dominantArm}`);
    console.log(`Backhand preference: ${backhand || 'not set'}`);
    console.log('Status: analysis started');
    jobId = await createAnalysisJob(req.user, {
      originalFilename: req.file.originalname,
      storedFilename: req.file.filename,
      dominantArm,
      backhand,
    });
    const analysis = await runAnalysis(req.file.path, { dominantArm, backhand });
    logResults(req.file.originalname, { dominantArm, backhand }, analysis);
    await saveAnalysisResult(jobId, analysis);
    const uploadedAt = new Date().toISOString();
    saveLatest({
      filename: req.file.originalname,
      storedAs: req.file.filename,
      uploadedAt,
      duration: analysis.duration,
      strokes: analysis.counts,
      comments: analysis.comments,
      overlay: analysis.overlay,
    });
    const comments = commentsForClient(analysis.comments);
    res.status(201).json({
      status: 'processed',
      jobId,
      filename: req.file.originalname,
      storedAs: req.file.filename,
      size: req.file.size,
      uploadedAt,
      duration: analysis.duration,
      dominantArm,
      backhand,
      strokes: analysis.counts,
      comments,
      overlay: analysis.overlay,
      strokesCsv: analysis.strokesCsv,
    });
  } catch (error) {
    await failAnalysisJob(jobId, error.message);
    console.error('--- Analysis failed ---');
    console.error(error.stderr || error.stdout || error.message);
    res.status(500).json({ error: 'Video analysis failed' });
  }
});

app.use((error, _req, res, _next) => {
  res.status(400).json({ error: error.message || 'Upload failed' });
});

app.listen(PORT, () => {
  console.log(`Upload server listening on http://localhost:${PORT}`);
});
