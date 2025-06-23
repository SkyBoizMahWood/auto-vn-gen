# auto-vn-gen

An automated Visual Novel Generation Algorithm using Large Language Models. This project generates complete visual novel storylines with branching paths, characters, scenes, and optional image generation.

## Setup

### Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose
- Windows 10/11 (for Unsloth acceleration - optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SkyBoizMahWood/auto-vn-gen.git
cd auto-vn-gen
```

2. Create a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
# For basic installation
pip install -r requirements.txt

# For Unsloth-accelerated version (Windows only)
pip install -r requirements_tuned_model.txt
```

4. Create `.env` file with required credentials:
```
NEO4J_AUTH=neo4j/your_password
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

5. Start the Neo4j database:
```bash
docker-compose up -d
```

### Using Unsloth Acceleration (Optional)

For Windows users who want to use Unsloth acceleration:
1. Install CUDA 12.1
2. Follow [Unsloth Windows installation guide](https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation)

## Usage

### Basic Story Generation

Generate a single story with default parameters:

```bash
python main.py generate-story
```

### Advanced Story Generation

Generate a story with custom parameters:

```bash
python main.py generate-story \
    --game-genre "psychological horror" \
    --themes "mystery,supernatural" \
    --num-chapters 4 \
    --num-endings 3 \
    --num-main-characters 6 \
    --num-main-scenes 7 \
    --enable-image-generation
```

### Batch Generation

Generate multiple stories:

```bash
python main.py batch-generation \
    --n-stories 5 \
    --enable-image-generation
```

### Available Parameters

- `--game-genre`: Type of visual novel (default: "visual novel")
- `--themes`: List of themes (optional)
- `--num-chapters`: Number of chapters (default: 3)
- `--num-endings`: Number of endings (default: 3)
- `--num-main-characters`: Number of main characters (default: 5)
- `--num-main-scenes`: Number of scenes/locations (default: 5)
- `--min-num-choices`: Minimum choices per opportunity (default: 2)
- `--max-num-choices`: Maximum choices per opportunity (default: 3)
- `--enable-image-generation`: Enable AI image generation (default: false)
- `--seed`: Random seed for reproducibility (optional)

### Utility Scripts

#### Regenerate Images
```bash
python -m scripts.regenerate-images [story_id] --for-characters --for-scenes
```

#### Calculate Story Costs
```bash
python -m scripts.calculate.py cost-per-story [story_id]
```

#### Delete Story
```bash
python -m scripts.prune.py --story-id [story_id]
```

## Output Structure

Generated stories are saved in the `outputs` directory with the following structure:

```
outputs/
└── [story_id]/
    ├── context.json      # Generation context and metadata
    ├── plot.json        # Story plot and structure
    ├── responses.json   # LLM responses
    └── histories.json   # Generation history
```

## Development

### Project Structure

```
auto-vn-gen/
├── src/                # Source code
├── scripts/            # Utility scripts
├── outputs/            # Generated stories
├── logs/              # Application logs
└── neo4j/             # Database files
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request