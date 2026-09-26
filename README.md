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
- Render with MongoDB Atlas

### Render deployment

This repository includes `render.yaml`, which configures Render to install the dependencies, import the MovieLens dataset during startup, and run Streamlit on Render's assigned port.

1. Create a MongoDB Atlas cluster.
2. Create a database user and copy the Atlas connection string. Replace the password placeholder with the real password and URL-encode special characters in it.
3. In Atlas, add Render's outbound IP range to Network Access. For a quick test, `0.0.0.0/0` allows access from all IPs, but a restricted production network is preferable.
4. In Render, choose **New > Blueprint**, connect the `kaustub-lade/movie-rec` GitHub repository, and apply `render.yaml`.
5. Set the secret `MONGODB_URI` environment variable in Render to the Atlas connection string. `DB_NAME` is already set to `movie_recommender`.
6. Deploy. The startup command runs `python -m scripts.load_data` and imports the CSV dataset into Atlas before starting Streamlit. It is idempotent, so restarts and later deploys do not create duplicates.
7. Open the Render service URL. Render checks the Streamlit health endpoint at `/_stcore/health`.

Do not set `MONGODB_URI` to `mongodb://localhost:27017/` on Render; that points to the Render container, not your computer.

### Netlify deployment

Netlify deploys static sites and serverless functions. It does not directly run a persistent Streamlit server or local MongoDB service, so this Streamlit repository produces a Netlify 404 because it has no `index.html` entry point. Use the Render URL as the live application URL, or rewrite the application as a Netlify-compatible frontend and API.

## Security Notes

- Keep `.env` private and never commit database credentials.
- Use a restricted MongoDB database user for hosted deployments.
- Restrict MongoDB Atlas network access to the deployment environment where practical.
- Change the default database name and credentials for production deployments.
