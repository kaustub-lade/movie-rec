# Movie Recommender System

A user-based collaborative filtering movie recommender built with Python, MongoDB, pandas, scikit-learn, and Streamlit.

The application uses cosine similarity between users' ratings to find similar users and generates movie recommendations from their ratings.

## Features

- MongoDB-backed movie and rating storage
- Idempotent MovieLens dataset importer
- User-user cosine similarity recommendations
- Already-rated movie filtering
- Dashboard statistics
- User rating profiles and genre preferences
- Similar-user analysis
- Rating and similarity visualizations
- Movie title search
- Automated unit and database-layer tests

## Requirements

- Python 3.11 or newer
- MongoDB Community Server running locally, or a MongoDB Atlas deployment
- Windows, macOS, or Linux

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/kaustub-lade/movie-rec.git
cd movie-rec
```

Create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS or Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Configuration

Create a local environment file from the template:

```powershell
Copy-Item .env.example .env
```

For macOS or Linux:

```bash
cp .env.example .env
```

The default configuration expects MongoDB at:

```text
mongodb://localhost:27017/
```

For MongoDB Atlas, update `MONGODB_URI` in `.env`. Do not commit `.env`; it may contain database credentials and is ignored by Git.

## Load the Dataset

Make sure MongoDB is running, then import the bundled MovieLens Latest Small dataset:

```bash
python -m scripts.load_data
```

The importer creates the required indexes and is safe to run more than once. It imports approximately 9,742 movies and 100,836 ratings.

Useful options:

```bash
python -m scripts.load_data --drop   # Clear existing collections first
python -m scripts.load_data --force  # Re-download the dataset archive
```

## Run the Application

Start the Streamlit interface:

```bash
streamlit run main.py
```

Open the local URL shown by Streamlit, normally:

```text
http://localhost:8501
```

## Verify the Environment

Run the built-in self-check to verify MongoDB, imported data, recommendations, and rated-movie filtering:

```bash
python main.py --check
```

## Run Tests

Run the complete test suite through the active Python interpreter:

```bash
python -m pytest -q
```

The pure tests do not require MongoDB. Integration tests run only when a reachable MongoDB instance is configured through `MONGO_TEST_URI` or `MONGODB_URI`.

## Project Structure

```text
main.py                 Application entry point and runtime self-check
app/
  analytics.py          Analytics calculations and Plotly figures
  config.py             Environment-based configuration
  database.py           MongoDB access layer
  recommender.py        Collaborative filtering engine
  ui.py                 Streamlit presentation layer
data/
  movies.csv            Local MovieLens data, ignored by Git
  ratings.csv           Local MovieLens data, ignored by Git
scripts/
  load_data.py          Dataset download, extraction, and import
 tests/                  Unit and database-layer tests
requirements.txt        Python dependencies
.env.example            Safe configuration template
```

## Deployment

The current application is designed to run as a persistent Streamlit service and requires a reachable MongoDB database. The local MongoDB service cannot be used by a cloud deployment.

Recommended deployment options:

- Streamlit Community Cloud with MongoDB Atlas
- Render or another service that supports long-running Python processes with MongoDB Atlas

Netlify does not directly run Streamlit applications or persistent MongoDB services. Deploying this application to Netlify would require rewriting the frontend and backend around Netlify-compatible functions or a separate hosted API.

## Security Notes

- Keep `.env` private and never commit database credentials.
- Use a restricted MongoDB database user for hosted deployments.
- Restrict MongoDB Atlas network access to the deployment environment where practical.
- Change the default database name and credentials for production deployments.
