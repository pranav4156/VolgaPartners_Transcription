# Audio Transcription Pipeline

This project takes an audio file and turns it into text with timestamps. You
give it a recording (in almost any format), and it gives you back the words that
were spoken, along with when each segment and each word was said. The results
come out as JSON, and also as SRT and VTT subtitle files.

Part 1, the working pipeline, is implemented and tested. Part 2 is a system design write-up, which
the brief asked for in writing rather than code, so it lives further down in this README.

---

## What it does

- Accepts an audio file (WAV, MP3, M4A, FLAC, OGG, and more).
- Transcribes the speech into text.
- Returns timestamps for every segment, and for every word.
- Handles any audio format by converting it first.
- Handles long recordings without running out of memory.

## Setup

You need Python 3.10 to 3.12 (PyTorch does not have 3.13 or 3.14 builds yet) and
ffmpeg installed on your system.

```bash
brew install ffmpeg
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to use it (two ways to give input)

### Way 1: record your own voice and transcribe it

You record straight from your microphone with ffmpeg, then run the script on
that file. To find your microphone number, run
`ffmpeg -f avfoundation -list_devices true -i ""`. On a MacBook the built-in mic
is usually number 1.

```bash
ffmpeg -f avfoundation -i ":1" my_voice.wav
python transcribe.py my_voice.wav --model small
cat out/my_voice.json
```

Press `q` when you are done speaking. You can record for as long as you like.

### Way 2: transcribe a downloaded or long audio file

Point it at any audio or video file you have, such as a podcast, a meeting
recording, or something you downloaded. Long files are split into chunks
automatically once they go past ten minutes. The `--max-chunk-seconds` option
lets you force chunking on a shorter file so you can see that part working.

```bash
python transcribe.py path/to/long_audio.mp3 --model small --max-chunk-seconds 30
cat out/long_audio.json
```

The language is detected automatically (Whisper understands around 99
languages). You can force a language with `--language <code>` if you want.

### A few more options

```bash
python transcribe.py call.mp3 --formats json,srt
python transcribe.py call.m4a --model large-v3
python transcribe.py call.wav --provider faster-whisper
python transcribe.py call.wav --no-word-timestamps
```

The first run downloads the model, which is then cached. On a laptop without a
GPU, start with `--model small` so it stays fast.

---

## How the code is built, and why

The code is split into small classes, each doing one job. The goal was to keep
every piece easy to swap out later, which is what makes it ready to grow.

Here is the flow of one file through the system:

```
transcribe.py            you run this; it reads your options and starts the pipeline
      |
      v
TranscriptionPipeline    the coordinator; runs the steps in order
      |
      |--> AudioProcessor     checks the file and converts it to a standard format
      |--> AudioChunker       splits long files into pieces
      |--> Transcriber        runs the speech-to-text model
      |--> Formatter          writes the result as JSON, SRT, or VTT
```

### The main pieces

- **AudioProcessor** takes care of all the different audio formats. Instead of
  writing special code for each format, it sends every file through ffmpeg once
  and converts it to a single standard shape: 16 kHz, mono, 16-bit WAV. This is
  exactly what the speech model expects. So the rest of the code never has to
  worry about MP3 versus M4A versus anything else. If a file has no audio in it,
  this is where it gets rejected with a clear error.

- **AudioChunker** deals with long recordings. If a file is short, it passes
  straight through. If it is long, it gets cut into smaller overlapping pieces.
  Each piece remembers where it sits in the original file, so after each piece is
  transcribed, the timestamps are shifted back to line up with the full
  recording and stitched together. This means a three-hour file never has to be
  loaded into memory all at once.

- **Transcriber** is the actual speech-to-text engine, and it is written as an
  interface with two interchangeable versions behind it. The default is
  WhisperX, which gives very accurate word-level timestamps. The second is
  faster-whisper, which is lighter and works on more machines. Because they both
  sit behind the same interface, switching between them is a single setting, and
  adding a third option later (say, a cloud API) means writing one new class and
  changing nothing else.

- **Formatter** writes the output. There is one version for JSON, one for SRT,
  and one for VTT. Adding a new output format, like plain text, would just be one
  more small class.

- **TranscriptionPipeline** is the coordinator. It does not do any of the heavy
  work itself. It is handed the engine and the other pieces, and it runs them in
  order. Because the pieces are handed to it rather than hard-wired, the whole
  pipeline can be tested with a fake engine, which is why the tests run in
  seconds without downloading a real model.

### Key choices and trade-offs

- **I used open-source models that run locally, not a paid API.** This keeps the
  project self-contained, with no API keys or per-minute costs, and it keeps the
  focus on the engineering rather than on calling someone else's service.

- **WhisperX is the default because it gives precise word timestamps**, which is
  useful for things like subtitles and search. The trade-off is that it is a bit
  heavier to install, so faster-whisper is there as a lighter fallback.

- **ffmpeg is the only audio decoder.** The engine is given audio that has
  already been converted, so there is just one decoding path in the whole system.
  This also avoids a common install headache where the decoder library has to
  match the exact ffmpeg version on the machine.

- **The device is detected automatically.** If there is an NVIDIA GPU, it uses it
  with a bigger model. If not, it falls back to the CPU with a smaller, faster
  model. Any of this can be overridden with a setting.

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

There are 14 tests, and they do not need a GPU or a downloaded model. The engine
is replaced with a fake one, while ffmpeg, the chunking, the timestamp
stitching, and all the output formats run for real. The tests cover converting a
stereo 44.1 kHz MP3 into a clean 16 kHz mono WAV, rejecting files that are not
audio, splitting a long file into correctly offset chunks, the JSON, SRT, and
VTT output, and a full run from start to finish including the stitching of
chunk timestamps.

---

## Part 2: building this for scale (design, not implemented as per the instructions)

The script above handles one file at a time on one machine, which is fine for
the assessment. But the real question is what happens when this becomes a
service with a large number of users, many of them uploading long audio files
like hour-long meetings or podcasts. A single script would quickly fall over.
Here is how I would build it to handle that.

### How would you handle concurrent uploads?

The main idea is to keep the API asynchronous and non-blocking: never make the
user wait while their file is transcribed, and never do the heavy work inside the
web request.

- When a file comes in, the API validates it, saves it, creates a job record,
  puts that job on a queue, and immediately returns a job ID with a 202 Accepted
  response. The API is stateless, which means it holds no job data of its own and
  can be scaled horizontally by simply running more copies behind a load balancer.
- For large files, the file never passes through the API server. The API hands the
  client a short-lived pre-signed upload URL and the client uploads straight into
  object storage, using resumable multipart uploads so a dropped connection does
  not mean starting over.
- Transcription runs on a separate pool of worker processes that pull jobs off the
  queue. Throughput scales horizontally: add more workers to handle more load. The
  API and the workers scale independently of each other.
- The queue gives us backpressure, which is a fancy word for a shock absorber. If
  a burst of five hundred uploads arrives at once, the jobs line up and the
  workers drain them at a steady rate instead of the system being overwhelmed. GPU
  workers are the scarce, expensive resource, so this smoothing matters, and the
  worker pool can autoscale based on how deep the queue is.
- Per-user rate limiting and a cap on in-flight jobs per user keep things fair, so
  one heavy user cannot starve everyone else.

### How would you store audio and transcripts?

Different kinds of data go to the store that fits them best, chosen for
durability and cost.

- **Raw audio and the output files (JSON, SRT, VTT) go into object storage** like
  Amazon S3, Google Cloud Storage, or MinIO. It is cheap, effectively unlimited,
  and highly durable (S3 is designed for eleven nines of durability), which is
  exactly what large media files need. Users download their results through
  short-lived pre-signed links, so the files never pass back through the API.
- **A database like Postgres holds the small, queryable metadata**: the job ID,
  its status, the detected language, the duration, which model was used, the
  timing, and the keys that point to the files in object storage. If a transcript
  is small it can live in the database too; if it is large it stays in object
  storage next to the audio.
- **Audio is the most expensive thing to keep**, so a retention and lifecycle
  policy deletes it or moves it to cheaper cold storage after a set period, while
  keeping the much smaller transcripts around for longer.
- Because voice recordings are personal data (PII), everything is encrypted at
  rest and in transit, access follows least-privilege (workers get scoped,
  temporary credentials rather than broad access), and user deletion requests are
  honoured so data can actually be removed.

### How do you retry or recover failed transcriptions?

The system is built to be fault-tolerant, so a job is never silently lost.

- Every job is a state machine: queued, then processing, then either done or
  failed. The current state, the attempt count, and the last error are persisted
  in the database, so progress survives a restart.
- Queues deliver at-least-once, meaning a job can be handed out more than once. To
  make that safe, each job is idempotent: it is keyed by a hash of the input file,
  so processing the same audio twice produces the same result and reuses the
  existing output instead of redoing the work. No duplicate transcripts, no wasted
  GPU time.
- Transient failures (a worker crash, an out-of-memory error, a model that failed
  to load) are retried automatically with exponential backoff and jitter, so a
  struggling system is not hit by a wave of synchronized retries.
- A visibility timeout covers worker death: if a worker picks up a job and then
  dies without finishing, the queue makes the job visible again after a timeout so
  another worker takes over. A reaper also requeues any job stuck in processing
  past a deadline.
- After a few failed attempts a job is moved to a dead-letter queue instead of
  vanishing, so it can be inspected and replayed. If a downstream dependency (say
  object storage) starts failing, a circuit breaker stops the workers from
  hammering it and gives it room to recover.

### How would you expose this as an API?

A small, well-behaved REST API, which FastAPI is a good fit for.

- `POST /transcriptions` to submit a file, or to get a pre-signed link for large
  uploads. It validates the request and replies right away with a job ID and a
  202 Accepted.
- `GET /transcriptions/{id}` to check the status of a job.
- `GET /transcriptions/{id}/result?format=json|srt|vtt` to get the finished
  transcript in the format you want.
- `GET /healthz` as a liveness and readiness probe, so the orchestrator knows when
  a copy is alive and ready to receive traffic.

Because transcription takes time, the user submits a job and then either polls for
the result, or we send a webhook the moment it is done so they do not have to keep
asking. The API is protected with authentication (API keys or OAuth), rate
limiting, and strict input validation (reject non-audio and oversized files
early), is versioned (for example `/v1`) so it can change without breaking
existing clients, and shuts down gracefully so in-flight work is not dropped
during a deploy. FastAPI produces an OpenAPI documentation page on its own. The
best part is that the worker behind this API runs the exact same pipeline from
Part 1, so the core transcription code does not change at all.

### Reliability and safety, in short

The design leans on a handful of well-known practices to stay reliable and safe:

- **Reliability:** a stateless API that scales horizontally, a durable queue that
  absorbs bursts (backpressure), idempotent jobs so retries are safe, at-least-once
  delivery with a visibility timeout so no job is lost, exponential backoff with
  jitter, a dead-letter queue for jobs that keep failing, and a circuit breaker
  around shaky dependencies. The pipeline also degrades gracefully: if WhisperX is
  unavailable it falls back to the lighter faster-whisper engine.
- **Observability:** structured logs, metrics such as queue depth, job latency and
  failure rate, and distributed tracing across the API and the workers, with
  alerting tied to clear SLOs (service-level objectives), so problems are caught
  before users feel them.
- **Safety and security:** all data encrypted at rest and in transit,
  least-privilege scoped credentials for the workers, strict input validation to
  reject bad or oversized files early, rate limiting to resist abuse, and a clear
  data-retention and deletion policy because the audio is personal data.

### The whole picture

```
  user ---------> API (stateless, many copies, autoscaled)
   |               |  saves a job, returns a job ID
   |               v
   |             Queue (SQS or Redis)
   |               |
  (large files)    v
  upload ----->  GPU worker pool (autoscaled based on how long the queue is)
  straight to      |  reads audio from object storage
  object storage   |  transcribes it (the Part 1 pipeline)
   |               |  writes the transcript back to object storage
   |               v  and the job status to the database
   |             Postgres (job and transcript metadata)
   |               ^
   user polls -----'  the API reads the status and hands back a download link
```

In short: the API stays light and just manages jobs, a queue absorbs bursts, a
pool of workers does the heavy lifting and grows or shrinks with demand, object
storage holds the big files, and the database keeps the small facts. Retries,
idempotency, and a dead-letter queue make sure nothing gets lost along the way.

---

## Limitations

- It does not label who is speaking (diarization). That was out of scope.
- Speed and accuracy depend on the model and the hardware. The largest model is
  slow on a CPU, so use a GPU or a smaller model for quick results.
- When long files are split, the overlap handling that removes duplicate words at
  the boundaries is a simple rule. Aligning across boundaries would be the next
  improvement.
