from flask import Flask, request, jsonify, render_template
import os
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

app = Flask(__name__)

DEFAULT_PAYOFF_MATRIX = {
    "建大廠": {"減少": -9000, "持平": 10000, "增加": 50000},
    "建中廠": {"減少": 11000, "持平": 32000, "增加": 32000},
    "建小廠": {"減少": 24000, "持平": 24000, "增加": 24000}
}

def calculate_decision(matrix, criterion, alpha=0.4, probs=None):
    results = {}
    details = {}
    best_option = ""
    regret_matrix = None 
    
    if criterion == "Maximax":
        best_value = -float('inf')
        for opt, states in matrix.items():
            val = max(states.values())
            results[opt] = val
            details[opt] = f"Max( {', '.join(map(str, states.values()))} ) = {val}"
            if val > best_value: best_value = val; best_option = opt
                
    elif criterion == "Maximin":
        best_value = -float('inf')
        for opt, states in matrix.items():
            val = min(states.values())
            results[opt] = val
            details[opt] = f"Min( {', '.join(map(str, states.values()))} ) = {val}"
            if val > best_value: best_value = val; best_option = opt
                
    elif criterion == "Laplace":
        best_value = -float('inf')
        for opt, states in matrix.items():
            val = sum(states.values()) / len(states)
            results[opt] = round(val, 2)
            details[opt] = f"( {' + '.join(map(str, states.values()))} ) / {len(states)} = {round(val, 2)}"
            if val > best_value: best_value = val; best_option = opt
                
    elif criterion == "Hurwicz":
        best_value = -float('inf')
        for opt, states in matrix.items():
            max_v = max(states.values())
            min_v = min(states.values())
            val = alpha * max_v + (1 - alpha) * min_v
            results[opt] = round(val, 2)
            details[opt] = f"({alpha} × {max_v}) + ({(1-alpha):.1f} × {min_v}) = {round(val, 2)}"
            if val > best_value: best_value = val; best_option = opt
                
    elif criterion == "Minimax_Regret":
        states_keys = list(list(matrix.values())[0].keys())
        col_max = {state: max(matrix[opt][state] for opt in matrix) for state in states_keys}
        
        regret_matrix = {}
        best_value = float('inf') 
        for opt, states in matrix.items():
            regret_matrix[opt] = {}
            for state, val in states.items():
                regret_matrix[opt][state] = col_max[state] - val
            
            max_regret = max(regret_matrix[opt].values())
            results[opt] = max_regret
            details[opt] = f"遺憾值: {list(regret_matrix[opt].values())} ➔ 取最大 = {max_regret}"
            if max_regret < best_value: best_value = max_regret; best_option = opt
            
    elif criterion == "EMV": 
        best_value = -float('inf')
        for opt, states in matrix.items():
            val = sum(states[state] * probs[state] for state in states)
            results[opt] = round(val, 2)
            calc_str = " + ".join([f"({states[s]} × {probs[s]})" for s in states])
            details[opt] = f"{calc_str} = {round(val, 2)}"
            if val > best_value: best_value = val; best_option = opt

    return results, details, best_option, best_value, regret_matrix

def generate_ai_analysis(best_option, criterion):
    analysis = {"style": "", "risk_warning": "", "guiding_question": ""}
    if criterion == "Maximax":
        analysis["style"] = "樂觀型決策：追求潛在最大報酬。"
        analysis["risk_warning"] = "風險提示：過度樂觀可能導致低需求時承受巨大虧損。"
        analysis["guiding_question"] = "引導式提問：「若高需求發生機率極低，你的決策是否改變？」"
    elif criterion == "Maximin":
        analysis["style"] = "保守型決策：極力避免最壞情況發生。"
        analysis["risk_warning"] = "風險提示：過度保守 → 錯失市場爆發時的獲利機會。"
        analysis["guiding_question"] = "引導式提問：「若公司目前現金流充裕且急需市佔率，應採用哪一準則？」"
    elif criterion == "Laplace":
        analysis["style"] = "無特定機率決策：假設未來發生機率均等。"
        analysis["risk_warning"] = "風險提示：若現實生活發生機率明顯不均等，此方法會產生嚴重誤差。"
        analysis["guiding_question"] = "引導式提問：「現實中，這三種情況發生的機率真的會剛好各三分之一嗎？」"
    elif criterion == "Hurwicz":
        analysis["style"] = "主觀權重決策：根據個人樂觀/悲觀程度給予權重。"
        analysis["risk_warning"] = "風險提示：高度依賴決策者主觀的 α 值，缺乏客觀數據支持。"
        analysis["guiding_question"] = "引導式提問：「不同的 α 值會如何改變最終選擇？這說明了什麼？」"
    elif criterion == "Minimax_Regret":
        analysis["style"] = "機會成本最小化：注重事後不後悔。"
        analysis["risk_warning"] = "風險提示：為了將遺憾最小化，可能導致選擇中庸方案，無法在特定市場中利潤最大化。"
        analysis["guiding_question"] = "引導式提問：「避免後悔是否等同於替公司賺取最多利潤？兩者的差異在哪？」"
    elif criterion == "EMV":
        analysis["style"] = "客觀數據導向：依據市場預測機率進行精算。"
        analysis["risk_warning"] = "風險提示：極度依賴機率的準確度。若前端市場調查錯誤，結果將全盤皆輸。"
        analysis["guiding_question"] = "引導式提問：「若『減少』的機率稍微提高 10%，你的決策是否會翻盤？」"
    return analysis

@app.route('/')
def index():
    return render_template('Capacity.html')

@app.route('/api/decision', methods=['POST'])
def decision_api():
    data = request.json
    criterion = data.get('criterion', 'Maximax')
    alpha = float(data.get('alpha', 0.4))
    matrix = data.get('matrix', DEFAULT_PAYOFF_MATRIX)
    probs = data.get('probs', {"減少": 0.3, "持平": 0.4, "增加": 0.3}) 
    
    results, details, best_option, best_value, regret_matrix = calculate_decision(matrix, criterion, alpha, probs)
    ai_analysis = generate_ai_analysis(best_option, criterion)
    
    return jsonify({
        "criterion_used": criterion,
        "calculated_values": results,
        "calculation_details": details,
        "recommended_option": best_option,
        "best_value": best_value,
        "ai_analysis": ai_analysis,
        "regret_matrix": regret_matrix 
    })

@app.route('/api/submit', methods=['POST'])
def submit_answer():
    try:
        data = request.json
        
        # 1. 設定 Google 授權範圍
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        
        # 2. 讀取金鑰 (雙向通用寫法)
        creds_path = '/etc/secrets/credentials.json'
        
        if not os.path.exists(creds_path):
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            creds_path = os.path.join(BASE_DIR, 'credentials.json')
            
        creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
        client = gspread.authorize(creds)
        
        # 3. 開啟雲端試算表
        sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1Ioq-W8woOPfoZf60r4hAiuNI1iGGMr_M36f48fw1kUw/edit?gid=0#gid=0").sheet1
        
        # 4. 取得台灣時間
        tw_tz = datetime.timezone(datetime.timedelta(hours=8))
        timestamp = datetime.datetime.now(tw_tz).strftime("%Y-%m-%d %H:%M:%S")
        
        # 5. 準備寫入的資料
        row_data = [
            timestamp, 
            data.get('studentClass', ''), 
            data.get('studentId', ''), 
            data.get('studentName', ''), 
            data.get('q1Answer', ''),         
            data.get('criterionUsed', ''),    
            data.get('aiQuestion', ''),       
            data.get('q2Answer', '')          
        ]
        
        # 6. 將資料新增到試算表的最後一行
        sheet.append_row(row_data)
        
        return jsonify({"status": "success", "message":""})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error", "message": "儲存失敗，請聯繫老師。"}), 500

@app.route('/admin')
def admin_view():
    return "<h2>資料已全數轉移至 Google 雲端試算表，請直接開啟試算表查看成績！</h2>"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
