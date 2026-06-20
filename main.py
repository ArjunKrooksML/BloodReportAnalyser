from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
# silence noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from fastapi import FastAPI
from app.api.routes import router
from app.db.store import init_db

log = logging.getLogger(__name__)
app = FastAPI(title="Blood Report Analyser")

@app.on_event("startup")
def startup():
    log.info("Starting up — initialising database")
    init_db()
    log.info("Database ready")

app.include_router(router, prefix="/api")
