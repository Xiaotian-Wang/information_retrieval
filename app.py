from flask import Flask, request
from flask_cors import CORS

from main import *

app = Flask(__name__)
CORS(app)

ysc, tyc = load_syn()


@app.route('/search', methods=['GET', 'POST'])
def search():
    res = {
        'code': 400,
        'data': {},
        'msg': 'failure'
    }
    try:
        data = dict(request.form)
        keyword = data['keyword']
        res['data']['result'] = rank_it(keyword, ysc=ysc, tyc=tyc, method=bm25)
        res['code'] = 200
        res['msg'] = 'success'
    except Exception as e:
        res['msg'] = str(e)

    return res


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=1996)
