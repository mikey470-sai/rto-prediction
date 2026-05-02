from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.main import app

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def serve_demo():
    return FileResponse("rto-demo.html")