from fastapi import APIRouter
from fastapi.responses import FileResponse

from src.database.crud import get_results

router = APIRouter()


@router.get("/results")
def results():

    data = get_results()

    return {"total_records": len(data), "results": data}


@router.get("/download/pdf")
def download_pdf():

    return FileResponse(
        path="export_results/evaluation_report.pdf",
        filename="evaluation_report.pdf",
        media_type="application/pdf",
    )


@router.get("/download/excel")
def download_excel():

    return FileResponse(
        path="export_results/evaluation_report.xlsx",
        filename="evaluation_report.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
