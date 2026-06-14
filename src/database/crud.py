from src.database.connection import get_connection


def save_result(question_id, pass1_marks, pass2_marks, final_marks):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
    INSERT INTO evaluation_results
    (
        question_id,
        pass1_marks,
        pass2_marks,
        final_marks
    )
    VALUES (?, ?, ?, ?)
    """,
        (question_id, pass1_marks, pass2_marks, final_marks),
    )

    conn.commit()

    conn.close()


def get_results():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("SELECT * FROM evaluation_results")

    results = cursor.fetchall()

    conn.close()

    return results
