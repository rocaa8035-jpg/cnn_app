"""TecnoForms · Lector de texto manuscrito manuscritos.
Uso:  pip install flask tensorflow opencv-python-headless pymupdf
      (coloca cnn_alfanumerico.keras junto a este archivo)  ->  python app.py  ->  http://localhost:5000
"""
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model("cnn_alfanumerico.keras")
LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def to_digit(crop):
    """Recorte binario -> 28x28 estilo MNIST (igual que en el cuaderno)."""
    h, w = crop.shape
    s = max(h, w)
    sq = np.zeros((s, s), np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    sq = cv2.resize(sq, (24, 24), interpolation=cv2.INTER_AREA)
    out = np.zeros((28, 28), np.uint8)
    out[2:26, 2:26] = sq
    return out.astype("float32") / 255.0


def read_page(bgr):
    scale = 1400 / max(bgr.shape[:2])
    if scale < 1:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15)
    H, W = th.shape
    for k in [(max(40, W // 15), 1), (1, max(40, H // 15))]:      # quita líneas de campos y tablas
        th = cv2.subtract(th, cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k)))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c) > 40 and max(15, 0.02 * H) <= h < 0.5 * H and 3 <= w < 0.5 * W:
            boxes.append((x, y, w, h))
    result, vis = [], bgr.copy()
    if boxes:
        med_h = float(np.median([b[3] for b in boxes]))
        boxes.sort(key=lambda b: b[1] + b[3] / 2)
        rows, cur = [], [boxes[0]]
        for b in boxes[1:]:                                       # agrupa en renglones
            if (b[1] + b[3] / 2) - np.mean([c[1] + c[3] / 2 for c in cur]) > 0.7 * med_h:
                rows.append(cur); cur = []
            cur.append(b)
        rows.append(cur)
        for row in rows:
            row.sort(key=lambda b: b[0])
            x_in = np.array([to_digit(th[y:y + h, x:x + w]) for x, y, w, h in row])[..., None]
            probs = model.predict(x_in, verbose=0)
            line = []
            for (x, y, w, h), p in zip(row, probs):
                d, conf = LABELS[int(p.argmax())], float(p.max())
                col = (80, 160, 0) if conf >= 0.9 else (0, 150, 240)
                cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
                cv2.putText(vis, str(d), (x, max(y - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                line.append({"d": d, "conf": round(conf, 4)})
            result.append(line)
    ok, png = cv2.imencode(".png", vis)
    return {"image": "data:image/png;base64," + base64.b64encode(png).decode(), "lines": result}


@app.post("/predict")
def predict():
    f = request.files.get("file")
    if not f:
        return jsonify(error="No se recibió ningún archivo."), 400
    data = f.read()
    try:
        if f.filename.lower().endswith(".pdf"):
            import fitz  # PyMuPDF
            pages = []
            for pg in list(fitz.open(stream=data, filetype="pdf"))[:10]:
                pix = pg.get_pixmap(dpi=150, alpha=False)
                arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
                pages.append(read_page(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)))
        else:
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return jsonify(error="El archivo no es una imagen ni un PDF válido."), 400
            pages = [read_page(img)]
        return jsonify(pages=pages)
    except Exception as e:
        return jsonify(error=f"No se pudo procesar el archivo: {e}"), 500


@app.get("/")
def index():
    return INDEX


INDEX = r"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TecnoForms · Lector de texto manuscrito</title>
<style>
:root{--bg:#f6f7fb;--card:#fff;--ink:#1c2233;--mut:#6b7385;--line:#e3e6ef;--brand:#4f46e5;--ok:#16a34a;--warn:#d97706}
@media(prefers-color-scheme:dark){:root{--bg:#0f1220;--card:#181c30;--ink:#eef0f8;--mut:#9aa2b8;--line:#2a3050;--brand:#818cf8;--ok:#4ade80;--warn:#fbbf24}}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
main{max-width:920px;margin:0 auto;padding:32px 20px 60px}h1{margin:0;font-size:28px}p.sub{color:var(--mut);margin:6px 0 24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px;margin-bottom:18px}
#drop{border:2px dashed var(--line);border-radius:16px;padding:44px 20px;text-align:center;cursor:pointer;transition:.15s}
#drop.on,#drop:hover{border-color:var(--brand);background:color-mix(in srgb,var(--brand) 8%,transparent)}
#drop b{font-size:18px}#drop span{display:block;color:var(--mut);margin-top:6px;font-size:14px}
.steps{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px;color:var(--mut);font-size:13px}.steps i{font-style:normal;background:var(--bg);border:1px solid var(--line);border-radius:99px;padding:4px 12px}
.row{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:14px}.row label{font-size:14px;color:var(--mut)}
button{background:var(--brand);color:#fff;border:0;border-radius:10px;padding:9px 16px;font-weight:600;cursor:pointer}button.g{background:transparent;color:var(--ink);border:1px solid var(--line)}
.line{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}.num{font:600 26px ui-monospace,Consolas,monospace;letter-spacing:.12em}
.chip{display:inline-flex;flex-direction:column;align-items:center;border:1px solid var(--line);border-radius:10px;padding:4px 10px;min-width:46px}
.chip b{font-size:20px}.chip small{font-size:11px;color:var(--ok)}.chip.low{border-color:var(--warn)}.chip.low small{color:var(--warn)}
img.res{width:100%;border-radius:12px;border:1px solid var(--line)}.err{color:#dc2626}.hide{display:none}
.spin{width:22px;height:22px;border:3px solid var(--line);border-top-color:var(--brand);border-radius:50%;animation:s .8s linear infinite;margin:auto}@keyframes s{to{transform:rotate(360deg)}}
</style></head><body><main>
<h1>Lector de texto manuscrito</h1>
<p class="sub">Sube una foto, un escaneo o un PDF de un formulario y la red neuronal leerá los números y letras escritos a mano.</p>
<div class="card"><div id="drop"><b>Arrastra tu archivo aquí o haz clic para elegirlo</b><span>JPG, PNG, WEBP o PDF · también puedes pegar una captura con Ctrl+V</span></div>
<input id="file" type="file" accept="image/*,.pdf" hidden>
<div class="steps"><i>1 · Sube el archivo</i><i>2 · Se detectan y recortan los caracteres</i><i>3 · La CNN los clasifica</i><i>4 · Revisa y copia el resultado</i></div></div>
<div id="busy" class="card hide"><div class="spin"></div></div><div id="err" class="card err hide"></div>
<div id="out" class="hide"><div class="card"><div class="row"><label>Marcar como dudosos los caracteres con confianza menor a
<b id="tv">90</b>%</label><input id="th" type="range" min="50" max="99" value="90"><button class="g" id="copy">Copiar texto</button><button class="g" id="csv">Descargar CSV</button></div><div id="pages"></div></div></div>
</main><script>
const $=id=>document.getElementById(id);let DATA=null;
const drop=$('drop'),inp=$('file');
drop.onclick=()=>inp.click();inp.onchange=()=>inp.files[0]&&send(inp.files[0]);
['dragover','dragenter'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('on')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('on')}));
drop.addEventListener('drop',ev=>ev.dataTransfer.files[0]&&send(ev.dataTransfer.files[0]));
addEventListener('paste',ev=>{const f=[...ev.clipboardData.files][0];f&&send(f)});
async function send(f){$('err').classList.add('hide');$('out').classList.add('hide');$('busy').classList.remove('hide');
 const fd=new FormData();fd.append('file',f,f.name||'captura.png');
 try{const r=await fetch('/predict',{method:'POST',body:fd});const j=await r.json();if(!r.ok)throw new Error(j.error);DATA=j.pages;render();$('out').classList.remove('hide')}
 catch(e){$('err').textContent=e.message;$('err').classList.remove('hide')}$('busy').classList.add('hide')}
function render(){const t=$('th').value/100;$('tv').textContent=$('th').value;let h='';
 DATA.forEach((p,i)=>{h+=`<h3>${DATA.length>1?'Página '+(i+1):'Resultado'}</h3><img class="res" src="${p.image}">`;
 if(!p.lines.length)h+='<p>No se detectaron caracteres. Prueba con una imagen más nítida y bien iluminada.</p>';
 p.lines.forEach((l,k)=>{h+=`<div class="line"><span style="color:var(--mut);font-size:13px">Línea ${k+1}</span><span class="num">${l.map(c=>c.d).join('')}</span></div><div class="line">`+
 l.map(c=>`<span class="chip ${c.conf<t?'low':''}"><b>${c.d}</b><small>${(c.conf*100).toFixed(1)}%</small></span>`).join('')+'</div>'})});
 $('pages').innerHTML=h}
$('th').oninput=()=>DATA&&render();
const txt=()=>DATA.flatMap(p=>p.lines.map(l=>l.map(c=>c.d).join(''))).join('\n');
$('copy').onclick=async()=>{await navigator.clipboard.writeText(txt());$('copy').textContent='¡Copiado!';setTimeout(()=>$('copy').textContent='Copiar texto',1500)};
$('csv').onclick=()=>{let s='pagina,linea,posicion,caracter,confianza\n';DATA.forEach((p,i)=>p.lines.forEach((l,k)=>l.forEach((c,j)=>s+=`${i+1},${k+1},${j+1},${c.d},${c.conf}\n`)));
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([s],{type:'text/csv'}));a.download='caracteres.csv';a.click()};
</script></body></html>"""

if __name__ == "__main__":
    app.run(debug=False, port=5000)
