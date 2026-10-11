/* DefectMaster Lite — 純 JS，無 React / Babel / Tailwind。
   記憶體策略：
     · 圖片清單分頁向後端要，畫面上只保留當頁 DOM
     · 縮圖走 /image?w=160 的伺服器端快取縮圖，不讓瀏覽器解原圖
     · 標註逐張讀寫，改一張存一張，瀏覽器不持有整份資料集 */
'use strict';

const $ = (id) => document.getElementById(id);
const PALETTE = ['#2563eb', '#059669', '#d97706', '#dc2626', '#7c3aed', '#0891b2',
                 '#db2777', '#65a30d', '#ea580c', '#4f46e5'];
const MIN_BOX_DIM_PX = 10;    // 換算 640×640 輸入後，寬或高低於這個像素數視為太小
const AI_COLOR = '#0ea5e9';   // AI 預標、尚未人工調整的框（與 style.css 的 --sky 同色）

const S = {
  page: 'dataset',
  ds: null,                 // 資料集摘要
  list: { items: [], total: 0, offset: 0, limit: 60, filter: 'all' },
  cur: null,                // { path, width, height, boxes }
  curIdx: -1,
  dirty: false,             // 有未儲存的修改
  hasBackup: false,         // 磁碟上有沒有已存檔的還原點
  undoCount: 0,             // 磁碟上還剩幾層可以退
  picker: 'native',         // 'native' = 系統對話框 / 'builtin' = 網頁內建瀏覽器
  hist: [],                 // 本機編輯歷程（尚未存檔的每一步）
  baseline: '[]',           // 磁碟上目前的標註，用來判斷有沒有真的改過
  sel: -1,                  // 選取中的框
  classIdx: 0,
  polling: null,
  infer: null,
  carrySrc: null,           // 上一張看過的框（給「沿用上一張」用）
  selMode: false,           // 縮圖清單是否在「選取」模式
  picked: new Set(),        // 選取模式下勾選的圖片路徑（只在目前這一頁內有效）
  lastBatchCount: 1,        // 最近一次移除操作涉及幾張圖（給「復原」按鈕決定要不要整批復原）
  aiBusy: false,            // AI 預標推論中（避免重複送出）
};

/* ---------------- 共用 ---------------- */
async function api(url, opts) {
  const res = await fetch(url, opts);
  let data = null;
  try { data = await res.json(); } catch (e) { data = null; }
  // 我們自己的 404 都有中文說明；只有 FastAPI 預設的 "Not Found" 代表這條路由不存在——
  // 幾乎都是「網頁已經是新版、但 server 還是舊程式在跑」，直接講清楚要重啟。
  if (res.status === 404 && data && data.detail === 'Not Found')
    throw new Error('伺服器還在執行舊版程式，請重新啟動 server（在終端機按 Ctrl+C，再執行 python3 server.py）');
  if (!res.ok) throw new Error((data && (data.detail || data.message)) || `HTTP ${res.status}`);
  return data;
}
async function post(url, body) {
  return api(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
}
let toastTimer = null;
function toast(msg, kind) {
  let el = document.querySelector('.toast');
  if (!el) { el = document.createElement('div'); document.body.appendChild(el); }
  el.className = 'toast' + (kind ? ' ' + kind : '');
  el.textContent = msg;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.remove(), 5200);
}

/* 自訂確認 / 輸入框，取代 window.confirm / window.prompt。
   原生 confirm 一旦被使用者勾「不再顯示對話框」，之後每次呼叫都會「立刻回傳 false 且不顯示」，
   讓存檔、切換圖片等所有 confirm 判斷靜默失效——所以改用自己畫的 modal。 */
const CF = { resolve: null, mode: 'confirm', key: '' };
function cfEnd(val) {
  $('cfModal').hidden = true;
  const r = CF.resolve; CF.resolve = null;
  if (r) r(val);
}
$('cfOk').onclick = () => {
  if (!$('cfDontAskWrap').hidden && $('cfDontAsk').checked && CF.key) {
    try { localStorage.setItem('dm_skip_' + CF.key, '1'); } catch (e) { /* 隱私模式等 */ }
  }
  cfEnd(CF.mode === 'prompt' ? ($('cfInput').value || '') : true);
};
$('cfCancel').onclick = () => cfEnd(CF.mode === 'prompt' ? null : false);
$('cfModal').onclick = e => { if (e.target === $('cfModal')) cfEnd(CF.mode === 'prompt' ? null : false); };
document.addEventListener('keydown', e => {
  if ($('cfModal').hidden) return;
  if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); cfEnd(CF.mode === 'prompt' ? null : false); }
  else if (e.key === 'Enter') { e.preventDefault(); e.stopPropagation(); $('cfOk').click(); }
}, true);

function cfCancelPending() {          // 若上一個對話框還開著，先把它當作取消收掉，避免 Promise 卡住
  if (CF.resolve) { const r = CF.resolve; CF.resolve = null; r(CF.mode === 'prompt' ? null : false); }
}

function confirmDialog(body, opts = {}) {
  const key = opts.dontAskKey || '';
  if (key) {
    try { if (localStorage.getItem('dm_skip_' + key) === '1') return Promise.resolve(true); } catch (e) { /* ignore */ }
  }
  cfCancelPending();
  CF.mode = 'confirm'; CF.key = key;
  $('cfTitle').textContent = opts.title || '確認';
  $('cfBody').textContent = body;
  $('cfInput').hidden = true;
  $('cfDontAskWrap').hidden = !key;
  $('cfDontAsk').checked = false;
  $('cfOk').textContent = opts.okText || '確定';
  $('cfCancel').textContent = opts.cancelText || '取消';
  $('cfModal').hidden = false;
  return new Promise(res => { CF.resolve = res; });
}

function promptDialog(body, def = '', opts = {}) {
  cfCancelPending();
  CF.mode = 'prompt'; CF.key = '';
  $('cfTitle').textContent = opts.title || '輸入';
  $('cfBody').textContent = body;
  $('cfDontAskWrap').hidden = true;
  $('cfInput').hidden = false;
  $('cfInput').value = def;
  $('cfOk').textContent = opts.okText || '確定';
  $('cfCancel').textContent = opts.cancelText || '取消';
  $('cfModal').hidden = false;
  setTimeout(() => { $('cfInput').focus(); $('cfInput').select(); }, 0);
  return new Promise(res => { CF.resolve = res; });
}

const colorOf = (name) => {
  // 沒開資料集時 S.ds 是 {open:false}、沒有 classes——以前這裡會丟例外，推論頁的框整個畫不出來。
  // 類別不在資料集裡（例如模型叫 no-helmet、資料集叫 no_helmet）就用名稱算一個固定顏色。
  const cls = (S.ds && S.ds.classes) || [];
  let i = cls.indexOf(name);
  if (i < 0) { i = 0; for (const ch of String(name)) i = (i * 31 + ch.charCodeAt(0)) >>> 0; }
  return PALETTE[i % PALETTE.length];
};
const stat = (label, value) => `<div class="stat"><b>${value}</b><span>${label}</span></div>`;

/* ---------------- 導覽 ---------------- */
document.querySelectorAll('.nav-item').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.nav-item').forEach(b => b.classList.toggle('active', b === btn));
    S.page = btn.dataset.page;
    document.querySelectorAll('.page').forEach(p => p.classList.toggle('active', p.id === 'page-' + S.page));
    $('crumb').textContent = btn.textContent.trim();
    if (S.page === 'label') openLabelPage();
    if (S.page === 'train') loadWeights();
    if (S.page === 'eval') { loadRuns(); loadTestEvalCard(); }
    if (S.page === 'infer') { loadModels(); refreshBxTarget(); }
  };
});

/* ---------------- 檔案瀏覽器（取代原生對話框：視窗大、可遠端用、樣式一致） -------- */
const FS = { kind: 'yaml', cur: '', sel: '', onPick: null };

function openFs(kind, title, startPath, onPick) {
  FS.kind = kind; FS.onPick = onPick; FS.sel = '';
  $('fsTitle').textContent = title;
  $('fsFilesHead').textContent = kind === 'image' ? '圖片檔' : kind === 'yaml' ? 'data.yaml 設定檔' : '檔案';
  $('fsChoose').disabled = true;
  $('fsSelected').textContent = kind === 'folder' ? '將選擇目前所在資料夾' : '尚未選擇';
  $('fsModal').hidden = false;
  fsGo(startPath || '');
}
function closeFs() { $('fsModal').hidden = true; FS.onPick = null; }
$('fsClose').onclick = closeFs;
$('fsModal').onclick = e => { if (e.target === $('fsModal')) closeFs(); };
document.addEventListener('keydown', e => { if (e.key === 'Escape' && !$('fsModal').hidden) closeFs(); });

async function fsGo(path) {
  try {
    const r = await api(`/fs/list?kind=${FS.kind}&path=${encodeURIComponent(path || '')}`);
    FS.cur = r.path;
    $('fsPath').value = r.path;
    $('fsUp').disabled = !r.parent;
    $('fsShortcuts').innerHTML = r.shortcuts.map(s =>
      `<button data-p="${encodeURIComponent(s.path)}">${s.name}</button>`).join('');
    $('fsShortcuts').querySelectorAll('button').forEach(b =>
      b.onclick = () => fsGo(decodeURIComponent(b.dataset.p)));
    $('fsCrumbs').innerHTML = '<span data-p="' + encodeURIComponent('/') + '">/</span>' +
      r.crumbs.map(c => `<span data-p="${encodeURIComponent(c.path)}">${c.name}</span>`).join('<b>/</b>');
    $('fsCrumbs').querySelectorAll('span').forEach(x =>
      x.onclick = () => fsGo(decodeURIComponent(x.dataset.p)));

    $('fsDirs').innerHTML = r.dirs.length ? r.dirs.map(d =>
      `<div class="fs-item" data-p="${encodeURIComponent(d.path)}"><span class="ic">▸</span><span class="nm">${d.name}</span></div>`
    ).join('') : '<div class="fs-empty">（沒有子資料夾）</div>';
    $('fsDirs').querySelectorAll('.fs-item').forEach(it =>
      it.onclick = () => fsGo(decodeURIComponent(it.dataset.p)));

    if (FS.kind === 'folder') {
      $('fsFiles').innerHTML = '<div class="fs-empty">資料夾模式：直接按「選擇」採用目前資料夾</div>';
      $('fsChoose').disabled = false;
      $('fsSelected').textContent = r.path;
      FS.sel = r.path;
    } else {
      $('fsFiles').innerHTML = r.files.length ? r.files.map(f =>
        `<div class="fs-item" data-p="${encodeURIComponent(f.path)}"><span class="ic">▪</span>
         <span class="nm">${f.name}</span><span class="sz">${(f.size / 1024).toFixed(0)} KB</span></div>`
      ).join('') : '<div class="fs-empty">這個資料夾沒有符合的檔案</div>';
      $('fsFiles').querySelectorAll('.fs-item').forEach(it => {
        const p = decodeURIComponent(it.dataset.p);
        it.onclick = () => {
          $('fsFiles').querySelectorAll('.fs-item').forEach(x => x.classList.remove('sel'));
          it.classList.add('sel');
          FS.sel = p; $('fsSelected').textContent = p; $('fsChoose').disabled = false;
        };
        it.ondblclick = () => { FS.sel = p; fsChoose(); };
      });
    }
  } catch (e) { toast(e.message, 'err'); }
}
function fsChoose() {
  if (!FS.sel) return;
  const cb = FS.onPick, v = FS.sel;
  closeFs();
  if (cb) cb(v);
}
$('fsChoose').onclick = fsChoose;
$('fsUp').onclick = () => fsGo(FS.cur.replace(/\/[^/]+\/?$/, '') || '/');
$('fsGo').onclick = () => fsGo($('fsPath').value.trim());
$('fsPath').addEventListener('keydown', e => { if (e.key === 'Enter') fsGo($('fsPath').value.trim()); });

/* ---------------- 資料集 ---------------- */
/** 挑檔案：預設走系統原生對話框（tkinter，視窗大小由 config.json 的 native_dialog_size 決定）。
    把 config.json 的 file_picker 改成 "builtin" 就會改用網頁內建的瀏覽器。 */
async function pickPath(kind, startPath, title, onPick) {
  if (S.picker === 'builtin') return openFs(kind, title, startPath, onPick);
  const url = kind === 'folder'
    ? `/pick-folder?start=${encodeURIComponent(startPath || '')}`
    : `/pick-file?kind=${kind}&start=${encodeURIComponent(startPath || '')}`;
  try {
    const r = await api(url);
    if (r.status === 'ok' && r.path) return onPick(r.path);
    if (r.status === 'unsupported') {           // 無桌面環境 → 自動退回內建瀏覽器
      toast(r.message || '無法開啟系統對話框，改用內建瀏覽器');
      return openFs(kind, title, startPath, onPick);
    }
    toast(r.message || '已取消選取');
  } catch (e) { toast('開啟選取視窗失敗：' + e.message, 'err'); }
}

$('btnPickYaml').onclick = () => pickPath('yaml', $('yamlPath').value.trim(),
  '選擇資料集的 data.yaml', p => { $('yamlPath').value = p; openYaml(); });
$('btnPickFolder').onclick = () => pickPath('folder', $('folderPath').value.trim(),
  '選擇圖片資料夾', p => { $('folderPath').value = p; openFolder(); });
$('btnOpenYaml').onclick = openYaml;
$('btnOpenFolder').onclick = openFolder;
$('yamlPath').addEventListener('keydown', e => { if (e.key === 'Enter') openYaml(); });
$('folderPath').addEventListener('keydown', e => { if (e.key === 'Enter') openFolder(); });

async function openYaml() {
  const p = $('yamlPath').value.trim();
  if (!p) return toast('請先選擇或輸入 data.yaml 路徑', 'err');
  await openDataset({ yaml: p });
}
async function openFolder() {
  const p = $('folderPath').value.trim();
  if (!p) return toast('請先選擇或輸入資料夾路徑', 'err');
  await openDataset({ folder: p });
}
/** {yaml} 或 {folder}：開啟資料集並同步輸入框（「回到原本的資料集」也走這裡）。 */
async function openDataset(src) {
  try {
    const d = await post('/dataset/open', src);
    if (src.yaml) $('yamlPath').value = src.yaml; else $('folderPath').value = src.folder;
    resetLabeling();                       // 換資料集 → 標註頁的清單/畫布一律重來
    renderDataset(d);
    toast(src.folder ? '資料夾已載入' : '資料集已載入', 'ok');
  } catch (e) { toast(e.message, 'err'); }
}

/* 換資料集後把標註頁清乾淨：舊清單、舊畫布、舊篩選都不能留，
   否則切到標註頁會看到上一個資料集的縮圖。 */
function resetLabeling() {
  S.list = { items: [], total: 0, offset: 0, limit: 60, filter: 'all' };
  S.cur = null; S.curIdx = -1; S.sel = -1;
  S.hasBackup = false; S.hist = []; S.baseline = '[]'; S.carrySrc = null; setDirty(false);
  S.selMode = false; S.picked.clear();
  $('btnUndo').disabled = true;
  $('btnCarryOver').disabled = true;
  $('btnSaveTrain').disabled = true;
  $('btnSaveTrain').textContent = '存到訓練資料夾';
  $('btnRestoreImg').hidden = true;
  $('btnUnTest').hidden = true;
  $('btnSelMode').textContent = '選取';
  $('selBar').hidden = true;
  $('filterSel').value = 'all';
  $('thumbs').innerHTML = '';
  $('pgInfo').textContent = '–';
  $('imgPos').textContent = '–';
  $('imgName').textContent = '（已切換資料集，請開啟標註工具）';
  $('saveState').textContent = '';
  cv.width = 0; cv.height = 0;
  $('lblTableBody').innerHTML = '<tr class="empty"><td colspan="4">尚未開啟資料集</td></tr>';
  $('lblCount').textContent = '–';
  if (S.page === 'label') loadPage(0);     // 正在標註頁 → 立刻換成新資料集
}

function renderDataset(d) {
  S.ds = d;
  if (!d || !d.open) { $('dsCard').hidden = true; mergeVisible(false); return; }
  $('dsCard').hidden = false;
  $('dsBadge').className = 'badge ok';
  const modeTxt = { yaml: 'data.yaml', merged: '已合併', split: '已切分' }[d.mode] || '資料夾';
  $('dsBadge').textContent = `${d.total} 張 · ${d.model_type} · ${modeTxt}`;
  $('dsStats').innerHTML = stat('總圖片', d.total) + stat('已標註', d.labeled) +
    stat('未標註', d.unlabeled) + stat('類別', d.classes.length) +
    (d.testset && d.testset.count
      ? `<div class="stat" title="${d.testset.dir}"><b>${d.testset.count}</b><span>測試資料夾</span></div>` : '') +
    `<div class="stat" style="flex:1"><b style="font-size:12px;word-break:break-all">${d.root}</b><span>資料集位置</span></div>`;
  const w = $('dsWarn');
  w.hidden = !(d.warnings && d.warnings.length);
  if (!w.hidden) w.innerHTML = d.warnings.map(x => '· ' + x).join('<br>');
  renderTrainsetNote(d);
  renderClasses();
  renderSplitStatus(d.split);
  mergeVisible(true);
  renderMergeSources(d.sources || [], d.consolidated_last);
  if (MG.scan) renderRemap(false);      // 類別可能因合併/移除而變動 → 重繪對應表的下拉選單
}

/* ---------------- 訓練資料夾（資料集頁）---------------- */
/** 一般資料集：顯示訓練資料夾的張數與 data.yaml，可一鍵切換過去訓練。
    開著的是訓練資料夾本身：提供「回到原本的資料集」，訓練開始後就能回去繼續修正。 */
function renderTrainsetNote(d) {
  const el = $('trainsetNote'), ts = d.trainset || {};
  let base = null;
  try { base = JSON.parse(lsGet('dm_prev_dataset') || 'null'); } catch (e) { base = null; }
  if (ts.is_trainset) {
    el.hidden = false;
    el.innerHTML = '<span>目前開著的是<b>訓練資料夾</b>，到「模型訓練」頁按開始訓練即可。訓練開始後可以回到原本的資料集繼續修正，不影響正在跑的訓練。</span>' +
      (base ? `<button class="btn ghost sm" id="btnBackToBase">回到原本的資料集</button><code>${base.yaml || base.folder}</code>` : '');
    if (base) $('btnBackToBase').onclick = () => openDataset(base);
    return;
  }
  el.hidden = !ts.count;
  if (!ts.count) return;
  el.innerHTML = `<span>訓練資料夾：<b>${ts.count}</b> 張（train ${ts.train} / val ${ts.val}）${ts.training ? ' · <b>訓練中</b>' : ''}</span>` +
    `<code>${ts.yaml}</code><button class="btn sm" id="btnOpenTrainset">開啟這個資料夾來訓練</button>`;
  $('btnOpenTrainset').onclick = () => openTrainset(d);
}

async function openTrainset(d) {
  const ts = d.trainset;
  if (!ts.train || !ts.val) return toast('訓練資料夾的 train 與 val 至少要各有一張圖，請再多加幾張', 'err');
  const ok = await confirmDialog(
    `切換到訓練資料夾（${ts.count} 張：train ${ts.train} / val ${ts.val}）？\n\n${ts.yaml}\n\n` +
    '切換後到「模型訓練」頁按開始訓練。訓練開始後，可以回資料集頁按「回到原本的資料集」繼續修正，不會影響正在跑的訓練。' +
    (d.base ? '' : '\n\n⚠ 目前的資料集套用了合併或切分，之後需要自己重新開啟原本的資料集（並重新套用）。') +
    unsavedNote(),
    { title: '開啟訓練資料夾', okText: '切換' });
  if (!ok) return;
  if (d.base) lsSet('dm_prev_dataset', JSON.stringify(d.base));
  await openDataset({ yaml: ts.yaml });
}


/* ---------------- 訓練／驗證／測試切分 ---------------- */
function updateSplitTrainPct() {
  const v = +$('splitVal').value || 0, t = +$('splitTest').value || 0;
  $('splitTrainPct').textContent = Math.max(0, 100 - v - t);
}
$('splitVal').oninput = updateSplitTrainPct;
$('splitTest').oninput = updateSplitTrainPct;

$('btnApplySplit').onclick = async () => {
  if (!S.ds || !S.ds.open) return toast('請先載入目前的資料集', 'err');
  const val_pct = +$('splitVal').value || 0, test_pct = +$('splitTest').value || 0;
  if (val_pct + test_pct >= 100) return toast('Val + Test 的百分比總和必須小於 100', 'err');
  const ok = await confirmDialog(
    '套用後訓練會自動改用這份切分產生的 data.yaml（原始資料與 data.yaml 不會被改動），' +
    '隨時可按「還原成原始切分」復原。\n\n確定套用嗎？',
    { title: '套用訓練切分', okText: '套用' });
  if (!ok) return;
  try {
    const r = await post('/dataset/split', { val_pct, test_pct });
    renderDataset(r.summary);
    toast(`已套用切分：Train ${r.split.counts.train}・Val ${r.split.counts.val}・Test ${r.split.counts.test}`, 'ok');
  } catch (e) { toast(e.message, 'err'); }
};

function renderSplitStatus(split) {
  const box = $('splitStatus');
  if (!split) { box.hidden = true; box.innerHTML = ''; return; }
  box.hidden = false;
  const c = split.counts;
  box.innerHTML = `<span>✓ 已套用切分（${split.created_at}）：Train ${c.train}・Val ${c.val}・Test ${c.test}` +
    `${c.test ? '' : '（Test 為 0，模型評估頁的「在測試集上評估」不會出現）'}</span>` +
    `<button class="btn ghost sm" id="btnRevertSplit">還原成原始切分</button>`;
  $('btnRevertSplit').onclick = async () => {
    if (!await confirmDialog('要撤銷目前套用的切分嗎？訓練資料來源會改回套用切分之前的狀態。',
      { title: '還原切分', okText: '還原' })) return;
    try {
      const r = await post('/dataset/split/revert', {});
      renderDataset(r.summary);
      toast('已還原', 'ok');
    } catch (e) { toast('還原失敗：' + e.message, 'err'); }
  };
}

function renderClasses() {
  const d = S.ds;
  $('classChips').innerHTML = d.classes.map((c, i) =>
    `<span class="chip"><i class="dot" style="background:${PALETTE[i % PALETTE.length]}"></i>${i}. ${c}
     <button data-act="rename" data-i="${i}">改名</button>
     <button data-act="del" data-i="${i}">刪除</button></span>`).join('') ||
    '<span class="hint">尚未建立類別，請在下方新增。</span>';
  refreshClassRestore();
  $('classChips').querySelectorAll('button[data-act=rename]').forEach(b => {
    b.onclick = async () => {
      const i = +b.dataset.i;
      const name = await promptDialog('新的類別名稱（只允許英數與底線）', S.ds.classes[i], { title: '改名' });
      if (!name) return;
      const next = S.ds.classes.slice();
      next[i] = name.replace(/[^A-Za-z0-9_-]/g, '');
      await saveClasses(next);
    };
  });
  $('classChips').querySelectorAll('button[data-act=del]').forEach(b => {
    b.onclick = () => deleteClass(+b.dataset.i);
  });
  const sel = $('classSel');
  sel.innerHTML = d.classes.map((c, i) => `<option value="${i}">${c}</option>`).join('');
  sel.value = String(Math.min(S.classIdx, Math.max(0, d.classes.length - 1)));
}
$('btnAddClass').onclick = async () => {
  const v = $('newClass').value.replace(/[^A-Za-z0-9_-]/g, '');
  if (!v) return;
  if (!S.ds || !S.ds.open) return toast('請先開啟資料集', 'err');
  await saveClasses(S.ds.classes.concat([v]));
  $('newClass').value = '';
};
/** 刪除類別：先問後端會影響多少，再讓使用者確認，最後整批重編號。 */
async function deleteClass(i) {
  const name = S.ds.classes[i];
  let u;
  try { u = await api('/dataset/classes/usage?index=' + i); }
  catch (e) { return toast(e.message, 'err'); }
  const msg = `確定要刪除類別「${i}. ${name}」嗎？\n\n` +
    `· 會刪掉 ${u.boxes} 個標註框\n` +
    `· 會改寫 ${u.files} 個標籤檔（後面的類別編號整批往前挪一號）\n` +
    `· 類別數會從 ${S.ds.classes.length} 變成 ${S.ds.classes.length - 1}\n\n` +
    `系統會先備份，之後可以按「復原上次刪除的類別」還原。`;
  if (!await confirmDialog(msg, { title: '刪除類別', okText: '刪除' })) return;
  try {
    const r = await post('/dataset/classes/delete', { index: i });
    S.ds.classes = r.classes;
    renderDataset(await api('/dataset/summary'));
    resetLabeling();
    toast(`已刪除「${name}」：移除 ${r.removed_boxes} 個框、改寫 ${r.touched_files} 個檔案`, 'ok');
  } catch (e) { toast('刪除失敗：' + e.message, 'err'); }
}

async function refreshClassRestore() {
  try {
    const d = await api('/dataset/classes/deleted');
    const b = $('btnRestoreClass');
    b.hidden = !d.has;
    if (d.has) b.textContent = `復原「${d.class}」（${d.files} 個檔案）`;
  } catch (e) { $('btnRestoreClass').hidden = true; }
}
$('btnRestoreClass').onclick = async () => {
  if (!await confirmDialog('要把上次刪除的類別連同所有標註一起還原嗎？', { title: '復原類別' })) return;
  try {
    const r = await post('/dataset/classes/restore');
    S.ds.classes = r.classes;
    renderDataset(await api('/dataset/summary'));
    resetLabeling();
    toast(`已復原類別「${r.class}」與 ${r.restored_files} 個標籤檔`, 'ok');
  } catch (e) { toast('復原失敗：' + e.message, 'err'); }
};

async function saveClasses(next) {
  try {
    const r = await post('/dataset/classes', { classes: next });
    S.ds.classes = r.classes;
    renderClasses();
    toast('類別已更新', 'ok');
  } catch (e) { toast(e.message, 'err'); }
}

/* ---------------- 合併外部資料集 ---------------- */
const MG = { scan: null, previewSig: null };

function mergeVisible(open) {
  $('mergeCard').hidden = !open;
  if (!open) {
    MG.scan = null; MG.previewSig = null;
    $('mergeBody').hidden = true; $('mergePreview').hidden = true;
  }
}

$('btnPickMergeYaml').onclick = () => pickPath('yaml',
  $('mergeYamlPath').value.trim() || (S.ds && S.ds.root) || '',
  '選擇要合併進來的外部 data.yaml', p => { $('mergeYamlPath').value = p; scanMerge(); });
$('btnScanMerge').onclick = scanMerge;
$('mergeYamlPath').addEventListener('keydown', e => { if (e.key === 'Enter') scanMerge(); });
$('btnRemapReset').onclick = () => renderRemap(true);
$('btnMergePreview').onclick = () => runMerge(true);
$('btnMergeRun').onclick = () => runMerge(false);
document.querySelectorAll('input[name=mergeSplit]').forEach(r => { r.onchange = invalidatePreview; });
$('mergeBg').onchange = invalidatePreview;
$('mergeRename').onchange = invalidatePreview;

async function scanMerge() {
  const p = $('mergeYamlPath').value.trim();
  if (!p) return toast('請先選擇外部 data.yaml', 'err');
  if (!S.ds || !S.ds.open) return toast('請先載入目前的資料集', 'err');
  try {
    const r = await api('/dataset/merge/scan?path=' + encodeURIComponent(p));
    MG.scan = r; MG.previewSig = null;
    const mismatch = r.task && r.current_task && r.task !== r.current_task;
    $('mergeSummary').innerHTML = stat('影像', r.images) + stat('外部類別', r.classes.length) +
      stat('有標籤', r.has_labels) + stat('任務', r.task) +
      stat('train/val', r.split_defined ? '已定義' : '未完整');
    const msgs = (r.warnings || []).slice();
    if (mismatch) msgs.unshift(`任務型別不符：外部為 ${r.task}，目前資料集為 ${r.current_task}，無法合併`);
    const w = $('mergeWarn');
    w.hidden = !msgs.length;
    w.innerHTML = msgs.map(x => '· ' + x).join('<br>');
    renderRemap(true);
    $('mergeBody').hidden = false;
    $('mergePreview').hidden = true;
    $('btnMergePreview').disabled = !!mismatch;
    $('btnMergeRun').disabled = true;
  } catch (e) { toast(e.message, 'err'); MG.scan = null; $('mergeBody').hidden = true; }
}

function autoMatchIdx(name) {
  if (!S.ds) return '__new__';
  const t = String(name).trim().toLowerCase();
  const i = S.ds.classes.findIndex(c => c.toLowerCase() === t);
  return i >= 0 ? String(i) : '__new__';
}

/** 依 MG.scan.classes + 目前資料集類別重建對應表。reset=true 走自動對應，false 盡量保留使用者已選的。 */
function renderRemap(reset) {
  if (!MG.scan || !S.ds) return;
  const prev = reset ? { mapping: {}, new_names: {} } : collectMapping();
  const opts = S.ds.classes.map((c, i) => `<option value="${i}">${i} · ${c}</option>`).join('');
  $('remapRows').innerHTML = MG.scan.classes.map(ec => {
    const nm = (prev.new_names[String(ec.idx)] || ec.name).replace(/"/g, '');
    const color = PALETTE[ec.idx % PALETTE.length];
    return `<div class="remap-row" data-ext="${ec.idx}">
      <span class="remap-src" title="${ec.name}"><i class="dot" style="background:${color}"></i>${ec.idx} · ${ec.name}</span>
      <span class="remap-arrow">→</span>
      <span class="remap-dst-cell">
        <select class="remap-dst">
          <option value="__new__">➕ 新增為新類別</option>
          <option value="__skip__">🚫 略過此類別</option>
          ${opts}
        </select>
        <input class="remap-newname" type="text" maxlength="40" value="${nm}" hidden>
      </span></div>`;
  }).join('');
  $('remapRows').querySelectorAll('.remap-row').forEach(row => {
    const ei = row.dataset.ext;
    const sel = row.querySelector('.remap-dst');
    const inp = row.querySelector('.remap-newname');
    let want = prev.mapping[ei] || autoMatchIdx(MG.scan.classes.find(c => String(c.idx) === ei).name);
    if (![...sel.options].some(o => o.value === want)) want = '__new__';   // 類別被回收後編號可能失效
    sel.value = want;
    const paint = () => {
      inp.hidden = sel.value !== '__new__';
      sel.classList.toggle('is-new', sel.value === '__new__');
      sel.classList.toggle('is-skip', sel.value === '__skip__');
    };
    paint();
    sel.onchange = () => { paint(); invalidatePreview(); };
    inp.oninput = () => { inp.value = inp.value.replace(/[^A-Za-z0-9_-]/g, ''); invalidatePreview(); };
  });
  invalidatePreview();
}

function collectMapping() {
  const mapping = {}, new_names = {};
  $('remapRows').querySelectorAll('.remap-row').forEach(row => {
    const ei = row.dataset.ext;
    const sel = row.querySelector('.remap-dst');
    mapping[ei] = sel.value;
    if (sel.value === '__new__') new_names[ei] = row.querySelector('.remap-newname').value.trim();
  });
  return { mapping, new_names };
}

function mergeReqBody(dry) {
  const { mapping, new_names } = collectMapping();
  return {
    yaml: $('mergeYamlPath').value.trim(), mapping, new_names, dry_run: dry,
    split: document.querySelector('input[name=mergeSplit]:checked').value,
    include_unlabeled: $('mergeBg').checked,
    dedup: $('mergeRename').checked ? 'rename' : 'skip',
  };
}
const previewSig = (b) => JSON.stringify(Object.assign({}, b, { dry_run: true }));

function invalidatePreview() {
  $('btnMergeRun').disabled = true;
  if (!$('mergePreview').hidden)
    $('mergePreview').innerHTML = '<pre>對應或選項已變更，請重新按「預覽合併結果」。</pre>';
}

async function runMerge(dry) {
  if (!MG.scan) return toast('請先讀取外部 yaml', 'err');
  const body = mergeReqBody(dry);
  if (!body.yaml) return toast('請先選擇外部 data.yaml', 'err');
  if (!dry) {
    if (MG.previewSig !== previewSig(body)) return toast('請先預覽這組對應的結果', 'err');
    if (!await confirmDialog('即將把外部影像整包複製進 DefectMaster_Output/.merged/ 並套用類別對應。\n' +
      '原始來源資料夾與其 data.yaml 不會被更動，合併後可用「移除此來源」還原。\n\n確定執行合併？',
      { title: '執行合併', okText: '合併' })) return;
  }
  $('btnMergePreview').disabled = true; $('btnMergeRun').disabled = true;
  try {
    const r = await post('/dataset/merge', body);
    const p = r.plan || {};
    if (dry) {
      const lines = [
        `將新增影像 .......... ${p.images_added}` + (p.duplicate ? `（另有 ${p.duplicate} 張因重複被略過）` : ''),
        `將寫出標籤檔 ........ ${p.labels_written}`,
        `重新編號的框 ........ ${p.boxes_remapped}`,
        `丟棄的框（略過類別）. ${p.boxes_dropped}`,
        `新增類別 ............ ${(p.classes_appended || []).join(', ') || '（無）'}`,
        `類別數 .............. ${p.class_count[0]} → ${p.class_count[1]}`,
        `略過的檔案 .......... 格式錯誤 ${p.files_malformed} · 無標籤檔 ${p.no_label}`,
        ...(p.notes || []).map(n => '· ' + n),
        '',
        '合併後訓練將改用產生的 .merged/_active/data.yaml（原 yaml 不變）',
      ];
      $('mergePreview').hidden = false;
      $('mergePreview').innerHTML = '<pre>' + lines.join('\n').replace(/</g, '&lt;') + '</pre>';
      MG.previewSig = previewSig(body);
      $('btnMergeRun').disabled = p.images_added === 0;
    } else {
      toast(`已合併：新增 ${p.images_added} 張影像` +
        ((p.classes_appended || []).length ? `、${p.classes_appended.length} 個新類別` : ''), 'ok');
      MG.scan = null; MG.previewSig = null;
      $('mergeBody').hidden = true; $('mergePreview').hidden = true; $('mergeYamlPath').value = '';
      renderDataset(await api('/dataset/summary'));
      resetLabeling();
    }
  } catch (e) { toast(e.message, 'err'); }
  finally { $('btnMergePreview').disabled = false; }
}

function renderMergeSources(sources, lastCons) {
  const box = $('mergeSources');
  if ((!sources || !sources.length) && !lastCons) { box.hidden = true; box.innerHTML = ''; return; }
  box.hidden = false;
  box.innerHTML = (sources.length ? '<div class="ms-title">已合併來源（暫存在 DefectMaster_Output/.merged/，原始資料夾沒有這些圖）</div>' : '') +
    sources.map(s =>
    `<div class="ms-row">
       <span>${s.origin}<br>
         <span class="ms-meta">+${s.images_added} 張 · ${(s.classes_added || []).length} 新類別 · ${s.split === 'train_val' ? 'train+val' : 'train'} · ${s.merged_at}</span>
       </span>
       <span class="row tight">
         <button class="btn sm" data-cons="${s.source_id}" data-n="${s.images_added}">併入目前資料夾</button>
         <button class="btn ghost sm" data-sid="${s.source_id}">移除此來源</button>
       </span>
     </div>`).join('') +
    (lastCons ? `<div class="ms-row"><span class="ms-meta">上次併入：${lastCons.source_id} · ${lastCons.count} 張 → ${lastCons.target} · ${lastCons.at}</span>
       <button class="btn ghost sm" id="btnUndoCons">復原上次併入</button></div>` : '');
  box.querySelectorAll('button[data-sid]').forEach(b => { b.onclick = () => removeSource(b.dataset.sid); });
  box.querySelectorAll('button[data-cons]').forEach(b => { b.onclick = () => consolidateSource(b.dataset.cons, +b.dataset.n); });
  if (lastCons) $('btnUndoCons').onclick = () => undoConsolidate(lastCons);
}

/** 併入目前資料夾：把暫存在 .merged/ 的圖片＋標籤真的複製進原始資料集的 train 資料夾。 */
async function consolidateSource(sid, n) {
  const ok = await confirmDialog(
    `把這個合併來源的 ${n} 張圖片與標籤，真的複製進目前資料集的 train 資料夾？\n\n` +
    '・檔名重複會自動加 _1、_2…，不會覆蓋既有檔案\n' +
    '・原始 data.yaml 的類別清單會補上合併時新增的類別\n' +
    '・完成後這個來源從「已合併來源」移除，.merged/ 裡的暫存複本會刪掉\n\n' +
    '之後重開原始資料集也看得到這些圖。可以用「復原上次併入」把複製進去的檔案刪掉。' + unsavedNote(),
    { title: '併入目前資料夾', okText: `複製 ${n} 張` });
  if (!ok) return;
  try {
    const r = await post('/dataset/merge/consolidate', { source_id: sid });
    renderDataset(r.summary);
    resetLabeling();
    toast(`已複製 ${r.copied} 張到 ${r.target}`, 'ok');
  } catch (e) { toast('併入失敗：' + e.message, 'err'); }
}

async function undoConsolidate(c) {
  const ok = await confirmDialog(
    `復原上次併入？\n\n會刪掉當時複製進 ${c.target} 的 ${c.count} 張圖片與它們的標籤檔` +
    '（若之後在標註頁修改過這些圖的標籤，修改也會一起刪掉）。\n\n外部原始資料夾沒有被動過，需要的話可以重新合併。',
    { title: '復原上次併入', okText: '刪除這些檔案' });
  if (!ok) return;
  try {
    const r = await post('/dataset/merge/consolidate/undo', {});
    renderDataset(r.summary);
    resetLabeling();
    toast(`已移除併入的 ${r.removed} 張`, 'ok');
  } catch (e) { toast('復原失敗：' + e.message, 'err'); }
}

async function removeSource(sid) {
  if (!await confirmDialog('要移除這個已合併來源嗎？\n\n會刪除 .merged/ 下複製的影像與重新編號的標籤，並盡量回收新增的類別。\n原始來源資料夾不受影響。', { title: '移除來源', okText: '移除' })) return;
  try {
    const r = await post('/dataset/merge/remove', { source_id: sid });
    const kept = (r.kept_classes || []).map(k => `${k.name}（${k.reason}）`);
    toast('已移除來源' +
      (r.removed_classes.length ? `，回收類別：${r.removed_classes.join(', ')}` : '') +
      (kept.length ? `；保留：${kept.join('、')}` : ''), 'ok');
    renderDataset(await api('/dataset/summary'));
    resetLabeling();
  } catch (e) { toast('移除失敗：' + e.message, 'err'); }
}

/* ---------------- 標註 ---------------- */
const cv = $('cv'), ctx = cv.getContext('2d');
const img = new Image();
let drag = null, view = { scale: 1, ox: 0, oy: 0 };

async function openLabelPage() {
  if (!S.ds || !S.ds.open) { toast('請先在「資料集」頁載入資料', 'err'); return; }
  if (S.labelFocus) {                      // 從推論頁「去標註」過來：翻到那張圖所在的頁並打開它
    const f = S.labelFocus.path;
    S.labelFocus = null;
    await loadPage(0, f);
  } else if (!S.list.items.length) await loadPage(0);
  refreshRemovedState();
  refreshTestsetState();
  loadAiModels();
}
$('filterSel').onchange = () => { S.list.filter = $('filterSel').value; loadPage(0); };
$('pgPrev').onclick = () => loadPage(Math.max(0, S.list.offset - S.list.limit));
$('pgNext').onclick = () => loadPage(S.list.offset + S.list.limit < S.list.total ? S.list.offset + S.list.limit : S.list.offset);

async function loadPage(offset, focus) {
  let q = `/dataset/images?offset=${offset}&limit=${S.list.limit}&filter=${S.list.filter}`;
  if (focus) q += '&focus=' + encodeURIComponent(focus);   // 後端會改成「那張圖所在的頁」的 offset
  const r = await api(q);
  offset = r.offset;
  S.list.items = r.items; S.list.total = r.total; S.list.offset = offset;
  S.picked.clear();          // 換頁／換篩選：選取狀態只在「這一頁」內有效
  updateSelBar();
  const from = r.total ? offset + 1 : 0;
  $('pgInfo').textContent = `${from}-${Math.min(offset + S.list.limit, r.total)} / ${r.total}`;
  renderThumbs();
  if (r.items.length) selectImage(focus && r.focus_index >= 0 ? r.focus_index : 0);
  else if (S.list.filter === 'infer_added') toast('目前資料集沒有從推論頁加入的圖片');
}

const TR_BADGE = '<b class="tr-badge" title="已在訓練資料夾">T</b>';

/** 重畫縮圖清單，不重新向後端要資料——選取模式切換/勾選時用這個，比 loadPage() 快。 */
function renderThumbs() {
  const box = $('thumbs');
  box.innerHTML = S.list.items.map((it, i) => {
    const picked = S.picked.has(it.path);
    return `<div class="thumb${picked ? ' picked' : ''}" data-i="${i}" title="${it.name}">
       ${S.selMode ? `<input type="checkbox" class="thumb-chk" ${picked ? 'checked' : ''}>` : ''}
       <img loading="lazy" src="/image?w=128&path=${encodeURIComponent(it.path)}" alt="">
       <span class="tname">${it.name}</span>
       ${it.labeled ? '<i></i>' : ''}
       ${it.in_train ? TR_BADGE : ''}
     </div>`;
  }).join('');
  box.querySelectorAll('.thumb').forEach(t => {
    const i = +t.dataset.i;
    t.onclick = S.selMode ? () => togglePick(i) : () => selectImage(i);
  });
}

function togglePick(i) {
  const it = S.list.items[i];
  if (!it) return;
  if (S.picked.has(it.path)) S.picked.delete(it.path); else S.picked.add(it.path);
  renderThumbs();
  updateSelBar();
}

function updateSelBar() {
  $('selPageN').textContent = S.list.items.length;
  $('selAll').checked = S.list.items.length > 0 && S.list.items.every(it => S.picked.has(it.path));
  $('btnDeleteSel').textContent = `刪除已選 (${S.picked.size})`;
  $('btnDeleteSel').disabled = S.picked.size === 0;
  $('btnTestSel').textContent = `移到測試資料夾 (${S.picked.size})`;
  $('btnTestSel').disabled = S.picked.size === 0;
}

/** 移除／移動圖片之後會重新載入清單，目前這張未存檔的修改會直接丟掉——確認視窗要先講清楚。 */
function unsavedNote() {
  return S.dirty ? '\n\n⚠ 目前這張圖有尚未儲存的標註修改，繼續的話會遺失。' : '';
}

function exitSelMode() {
  S.picked.clear();
  S.selMode = false;
  $('btnSelMode').textContent = '選取';
  $('selBar').hidden = true;
}

$('btnSelMode').onclick = () => {
  S.selMode = !S.selMode;
  $('btnSelMode').textContent = S.selMode ? '取消選取' : '選取';
  $('selBar').hidden = !S.selMode;
  if (!S.selMode) S.picked.clear();
  renderThumbs();
  updateSelBar();
};
$('selAll').onchange = () => {
  if ($('selAll').checked) S.list.items.forEach(it => S.picked.add(it.path));
  else S.picked.clear();
  renderThumbs();
  updateSelBar();
};
$('btnDeleteSel').onclick = async () => {
  const paths = [...S.picked];
  if (!paths.length) return;
  const ok = await confirmDialog(
    `確定要刪除選取的 ${paths.length} 張圖片嗎？\n\n圖片與標籤檔會搬到 DefectMaster_Output/.removed_images/，不會出現在訓練資料裡，可按「復原整批」還原。` +
    unsavedNote(),
    { title: '刪除已選圖片', okText: `刪除 ${paths.length} 張` });
  if (!ok) return;
  const keepIdx = S.curIdx;
  try {
    const r = await post('/dataset/image/remove_bulk', { paths });
    exitSelMode();
    await afterImagesRemoved(keepIdx);
    toast(`已刪除 ${r.removed} 張` +
      (r.skipped.length ? `，${r.skipped.length} 張已經不在磁碟上` : '') +
      '，可按「復原整批」還原', 'ok');
  } catch (e) { toast('刪除失敗：' + e.message, 'err'); }
};

/** 移到測試圖片資料夾：不是刪除，是把圖（連同標籤）搬出訓練資料，留給之後評估模型用；可整批移回。
    單張（工具列「移到測試資料夾」）與整批（選取模式）共用：先確認，再搬移並重新整理清單。 */
async function moveToTestset(paths, unlabeled) {
  if (!paths.length) return;
  let dir = '';
  try { dir = (await api('/dataset/testset')).dir; } catch (e) { /* 只是顯示用 */ }
  const one = paths.length === 1;
  const ok = await confirmDialog(
    `把${one ? '這張圖片' : `選取的 ${paths.length} 張圖片`}移到測試圖片資料夾？\n\n` +
    (one ? paths[0] + '\n→ ' : '') + `${dir || 'DefectMaster_Output/test_images/'}\n\n` +
    '圖片與標籤檔會一起搬過去，之後不會出現在標註清單、也不會參與訓練（不是刪除，可按「移回訓練集」搬回原位）。' +
    '可以在「模型評估」頁用這些圖評估模型。' +
    (unlabeled ? `\n\n⚠ ${one ? '這張' : `其中 ${unlabeled} 張`}還沒標註：評估時只會計入已標註的圖，建議先標好再移。` : '') +
    unsavedNote(),
    { title: '移到測試圖片資料夾', okText: one ? '移動' : `移動 ${paths.length} 張` });
  if (!ok) return;
  const keepIdx = S.curIdx;
  try {
    const r = await post('/dataset/testset/move', { paths });
    if (S.selMode) exitSelMode();       // 要在重繪縮圖之前關掉，否則縮圖還會帶著勾選框
    await afterImagesRemoved(keepIdx);
    toast(`已移動 ${r.moved} 張到測試圖片資料夾（目前共 ${r.testset.count} 張）：${r.testset.dir}` +
      (r.skipped.length ? `；${r.skipped.length} 張已經不在磁碟上` : ''), 'ok');
  } catch (e) { toast('移動失敗：' + e.message, 'err'); }
}
$('btnTestSel').onclick = () => moveToTestset([...S.picked],
  S.list.items.filter(it => S.picked.has(it.path) && !it.labeled).length);
$('btnTestImg').onclick = () => {
  if (!S.cur) return;
  const it = S.list.items[S.curIdx];
  moveToTestset([S.cur.path], it && !it.labeled ? 1 : 0);
};

async function selectImage(i) {
  const it = S.list.items[i];
  if (!it) return;
  if (S.dirty && !await confirmDialog('這張圖有尚未儲存的標註修改，離開就會遺失。\n\n確定要切換到其他圖片嗎？', { title: '尚未儲存', okText: '不儲存並切換' })) return;
  // 記下剛才那張的框，給「沿用上一張」按鈕用（連續影格常常框幾乎沒動）
  if (S.cur) {
    S.carrySrc = S.cur.boxes.filter(b => !b.readonly)
      .map(b => ({ className: b.className, x: b.x, y: b.y, w: b.w, h: b.h }));
  }
  $('btnCarryOver').disabled = !S.carrySrc || !S.carrySrc.length;
  S.curIdx = i; S.sel = -1; setDirty(false);
  document.querySelectorAll('.thumb').forEach(t => t.classList.toggle('sel', +t.dataset.i === i));
  $('imgName').textContent = it.path;
  $('imgPos').textContent = `${S.list.offset + i + 1} / ${S.list.total}`;
  const d = await api('/dataset/labels?path=' + encodeURIComponent(it.path));
  S.cur = { path: it.path, width: d.width, height: d.height, boxes: d.boxes };
  updateTrainBtn();
  S.hist = [];
  S.baseline = JSON.stringify(d.boxes);
  renderLabelTable();      // 不必等圖片載完就能先顯示清單；img.onload 之後 draw() 還會再同步一次
  img.onload = draw;
  img.src = '/image?path=' + encodeURIComponent(it.path);
  refreshBackupState();
}

/** 「沿用上一張」按鈕：手動把上一張看過的框帶進目前這張，只在按下時套用一次。
    這張圖已經有框的話，先問過再覆蓋；沒有的話直接套用。 */
async function useCarryOver() {
  if (!S.cur || !S.ds) return;
  if (!S.carrySrc || !S.carrySrc.length) return toast('目前沒有可沿用的上一張標註', 'err');
  if (S.cur.boxes.length) {
    const ok = await confirmDialog(
      `這張圖片目前有 ${S.cur.boxes.length} 個標註框。\n\n沿用上一張會覆蓋掉它（存檔前仍可按「復原」退回覆蓋前的內容）。\n\n確定要覆蓋嗎？`,
      { title: '沿用上一張會覆蓋現有標註', okText: '覆蓋', cancelText: '取消' });
    if (!ok) return;
  }
  const W = S.cur.width, H = S.cur.height;
  const carried = S.carrySrc
    .filter(b => S.ds.classes.includes(b.className))
    .map(b => {
      const x = Math.max(0, Math.min(b.x, W - 1)), y = Math.max(0, Math.min(b.y, H - 1));
      return { className: b.className, x, y,
               w: Math.max(1, Math.min(b.w, W - x)), h: Math.max(1, Math.min(b.h, H - y)) };
    });
  if (!carried.length) return toast('上一張沒有可帶入的類別（可能都已被刪除）', 'err');
  pushHist();
  S.cur.boxes = carried;
  S.sel = -1;
  syncDirty();
  draw();
  toast(`已沿用上一張 ${carried.length} 個框，確認後請儲存`, 'ok');
}
$('btnCarryOver').onclick = useCarryOver;

/* ---- 存到訓練資料夾：改好的圖「複製」成獨立的 YOLO 資料集，先拿去訓練一版 ---- */
const TRAIN_BTN_TIP = $('btnSaveTrain').title;
/** 目前這張已經在訓練資料夾裡就顯示 ✓（再按一次只是重新同步）；開著的就是訓練資料夾本身則停用。 */
function updateTrainBtn() {
  const b = $('btnSaveTrain'), it = S.list.items[S.curIdx];
  const isTs = !!(S.ds && S.ds.trainset && S.ds.trainset.is_trainset);
  b.disabled = !S.cur || isTs;
  b.title = isTs ? '目前開著的就是訓練資料夾本身' : TRAIN_BTN_TIP;
  b.textContent = it && it.in_train ? '✓ 已在訓練資料夾' : '存到訓練資料夾';
}

/** 有未存檔的修改就先走一般存檔流程（太小框、覆寫確認都照舊）；這張沒有任何框的話，
    第一次加入前先確認要當成「沒有目標的背景圖」。之後一般存檔會自動同步到訓練資料夾。 */
async function saveToTrainset() {
  if (!S.cur) return;
  if (S.dirty) {
    await saveCurrent();
    if (S.dirty) return;                       // 存檔確認被取消
  }
  const path = S.cur.path, idx = S.curIdx, it = S.list.items[idx];
  const background = !S.cur.boxes.some(b => !b.readonly);
  if (background && !(it && it.in_train) && !await confirmDialog(
    '這張圖沒有任何標註框，會當成「沒有目標的背景圖」加入訓練資料夾。\n\n確定要加入嗎？',
    { title: '加入背景圖', okText: '加入' })) return;
  try {
    const r = await post('/dataset/trainset/add', { path, background });
    if (it && it.path === path) {
      it.in_train = true;
      const t = document.querySelector(`.thumb[data-i="${idx}"]`);
      if (t && !t.querySelector('.tr-badge')) t.insertAdjacentHTML('beforeend', TR_BADGE);
    }
    if (S.cur && S.cur.path === path) updateTrainBtn();
    renderDataset(await api('/dataset/summary'));
    const ts = r.trainset;
    toast(`${r.added ? '已加入' : '已更新'}訓練資料夾（${r.split}）· 目前共 ${ts.count} 張：train ${ts.train} / val ${ts.val}`, 'ok');
  } catch (e) { toast('加入訓練資料夾失敗：' + e.message, 'err'); }
}
$('btnSaveTrain').onclick = saveToTrainset;

/* ---- AI 預標：用已訓練好的模型先把框畫出來，人工只要檢查、修正 ---- */
function lsGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
function lsSet(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* 隱私模式等 */ } }

/** 模型清單只列自己訓練的 .pt 與 engine——官方權重是 COCO 類別，跟資料集的類別對不上。 */
async function loadAiModels() {
  const sel = $('aiModel');
  let m;
  try { m = await api('/models'); } catch (e) { return; }
  const engines = (m.engines || []).map(e => e.name);
  const all = [...m.models, ...engines];
  const keep = sel.value || lsGet('dm_ai_model');
  sel.innerHTML =
    (m.models.length ? `<optgroup label="我的模型（.pt）">${m.models.map(x => `<option>${x}</option>`).join('')}</optgroup>` : '') +
    (engines.length ? `<optgroup label="TensorRT engine">${engines.map(x => `<option>${x}</option>`).join('')}</optgroup>` : '');
  sel.value = all.includes(keep) ? keep : (all[0] || '');
  $('btnAiAssist').disabled = !all.length || S.aiBusy;
  if (!all.length) $('btnAiAssist').title = '還沒有訓練好的模型，請先到「訓練」頁訓練一個';
}
$('aiModel').onchange = () => lsSet('dm_ai_model', $('aiModel').value);
$('aiConf').oninput = () => {
  $('aiConfVal').textContent = (+$('aiConf').value).toFixed(2);
  lsSet('dm_ai_conf', $('aiConf').value);
};
(() => {                                   // 還原上次用的信心門檻
  const c = parseFloat(lsGet('dm_ai_conf'));
  if (c >= 0.05 && c <= 0.95) { $('aiConf').value = String(c); $('aiConfVal').textContent = c.toFixed(2); }
})();

/** 動過（移動、改大小、換類別）的 AI 框視為已人工確認，拿掉 🤖 標記。標記只存在前端，不會寫進 txt。 */
function markReviewed(b) { delete b.source; delete b.conf; }

/** 「AI 預標」按鈕：只在按下時對目前這張圖跑一次推論，結果當成未存檔的編輯放進畫布。
    這張圖已經有框的話，先問過再覆蓋（唯讀框保留）；絕不自動存檔。 */
async function useAiAssist() {
  if (!S.cur || !S.ds || S.aiBusy) return;
  const model = $('aiModel').value;
  if (!model) return toast('還沒有訓練好的模型，請先到「訓練」頁訓練一個', 'err');
  const path = S.cur.path, conf = +$('aiConf').value;
  const btn = $('btnAiAssist');
  S.aiBusy = true; btn.disabled = true; btn.textContent = 'AI 偵測中…';
  let r;
  try {
    r = await post('/predict', { model, path, conf });
    if (!r.ok) throw new Error(r.error || '推論失敗');
  } catch (e) {
    return toast('AI 預標失敗：' + e.message, 'err');
  } finally {
    S.aiBusy = false; btn.disabled = false; btn.textContent = '🤖 AI 預標';
  }
  if (!S.cur || S.cur.path !== path) return;        // 等待期間換到別張了 → 結果作廢，不要套到錯的圖上
  const W = S.cur.width, H = S.cur.height;
  const skipped = new Set(), found = [];
  for (const d of r.dets) {
    if (d.type !== 'bbox') continue;
    if (!S.ds.classes.includes(d.cls)) { skipped.add(d.cls); continue; }
    const x = Math.max(0, Math.min(d.x, W - 1)), y = Math.max(0, Math.min(d.y, H - 1));
    found.push({ className: d.cls, x, y, w: Math.max(1, Math.min(d.w, W - x)), h: Math.max(1, Math.min(d.h, H - y)),
                 source: 'ai', conf: d.conf });
  }
  const skipMsg = skipped.size ? `；類別 ${[...skipped].join('、')} 不在這個資料集裡，已略過` : '';
  if (!found.length) return toast(`AI 沒有偵測到可用的框（信心門檻 ${conf.toFixed(2)}）${skipMsg}`, skipped.size ? 'err' : '');
  const editable = S.cur.boxes.filter(b => !b.readonly).length;
  if (editable) {
    const ok = await confirmDialog(
      `AI 偵測到 ${found.length} 個框，這張圖片目前有 ${editable} 個標註框。\n\n套用 AI 預標會覆蓋掉它們（存檔前仍可按「復原」退回覆蓋前的內容）。\n\n確定要覆蓋嗎？`,
      { title: 'AI 預標會覆蓋現有標註', okText: '覆蓋', cancelText: '取消' });
    if (!ok || !S.cur || S.cur.path !== path) return;
  }
  pushHist();
  S.cur.boxes = [...S.cur.boxes.filter(b => b.readonly), ...found];
  S.sel = -1;
  syncDirty();
  draw();
  toast(`AI 預標了 ${found.length} 個框（藍色虛線 🤖），請檢查修正後再儲存${skipMsg}`, 'ok');
}
$('btnAiAssist').onclick = useAiAssist;

async function refreshBackupState() {
  if (!S.cur) return;
  try {
    const b = await api('/dataset/labels/backup?path=' + encodeURIComponent(S.cur.path));
    S.hasBackup = !!b.has;
    S.undoCount = b.count || 0;
    $('btnUndo').textContent = S.undoCount > 1 ? `復原 (${S.undoCount})` : '復原';
    updateUndoBtn(b.saved_at);
  } catch (e) { S.hasBackup = false; S.undoCount = 0; updateUndoBtn(); }
}

/** 復原鈕：有未存檔修改可以丟棄，或磁碟上還有還原點可以退——兩者其一就能按。 */
function updateUndoBtn(savedAt) {
  const can = S.dirty || S.hist.length > 0 || S.hasBackup;
  $('btnUndo').disabled = !can;
  $('btnUndo').title = (S.dirty || S.hist.length) ? '退回上一步編輯（Ctrl+Z）'
    : S.hasBackup ? `退回上一個版本（${savedAt || '存檔前'}），還有 ${S.undoCount} 層`
    : '沒有可還原的內容';
}

/** 動手改之前先存一份快照，Ctrl+Z 才有東西可退。 */
function pushHist() {
  if (!S.cur) return;
  S.hist.push(JSON.stringify(S.cur.boxes));
  if (S.hist.length > 50) S.hist.shift();
}

/** 內容是否和磁碟上的不一樣——比對內容而不是設旗標，退回原狀時就會自動變回「已存檔」。 */
function syncDirty() {
  setDirty(S.cur ? JSON.stringify(S.cur.boxes) !== S.baseline : false);
}

function setDirty(v) {
  S.dirty = v;
  $('btnSave').disabled = !v;
  $('btnSave').classList.toggle('dirty', v);
  $('saveState').textContent = v ? '尚未儲存' : $('saveState').textContent;
  $('saveState').style.color = v ? 'var(--amber)' : 'var(--green)';
  updateUndoBtn();
}
$('prevImg').onclick = () => { if (S.curIdx > 0) selectImage(S.curIdx - 1); else if (S.list.offset) loadPage(S.list.offset - S.list.limit); };
$('nextImg').onclick = () => {
  if (S.curIdx < S.list.items.length - 1) selectImage(S.curIdx + 1);
  else if (S.list.offset + S.list.limit < S.list.total) loadPage(S.list.offset + S.list.limit);
};

/** 統一的「選類別」動作：更新下拉選單與之後新框的預設類別；若目前正選取一個可編輯的框，
    就一併把那個框改成這個類別（寫入歷程、標記未儲存）。下拉選單 onchange 與數字鍵 1-9 共用。 */
function setClass(idx) {
  if (!S.ds || !(idx >= 0) || idx >= S.ds.classes.length) return;
  S.classIdx = idx;
  $('classSel').value = String(idx);
  const b = (S.cur && S.sel >= 0) ? S.cur.boxes[S.sel] : null;
  if (b && !b.readonly && b.className !== S.ds.classes[idx]) {
    pushHist();
    b.className = S.ds.classes[idx];
    markReviewed(b);
    syncDirty();
    draw();
  }
}
$('classSel').onchange = () => setClass(+$('classSel').value || 0);

/** 這個框換算成 640×640 YOLO 輸入後佔的面積百分比。這個比例對「整張圖等比縮放到任何正方形
    輸入」都不變（框跟圖同比例縮放，比例會互相抵銷），所以不必真的算縮放，直接用原圖像素比即可。 */
function areaPct(b) {
  if (!S.cur || !S.cur.width || !S.cur.height) return 0;
  return (b.w * b.h) / (S.cur.width * S.cur.height) * 100;
}

/** 這個框換算成 640×640 YOLO 輸入後的寬高（px）。跟面積百分比不同，寬高要分別用 x／y
    兩軸各自的縮放倍率換算——圖片長寬比不是正方形時，兩軸縮放倍率並不相同。 */
function boxDims640(b) {
  if (!S.cur || !S.cur.width || !S.cur.height) return { w: 0, h: 0 };
  return { w: b.w * 640 / S.cur.width, h: b.h * 640 / S.cur.height };
}

/** 換算 640×640 後寬或高小於 MIN_BOX_DIM_PX，就視為太小、不建議標註（唯讀框不算）。 */
function isTooSmall(b) {
  if (b.readonly) return false;
  const d = boxDims640(b);
  return d.w < MIN_BOX_DIM_PX || d.h < MIN_BOX_DIM_PX;
}

/** 標註清單：跟 draw() 一起重繪，永遠反映目前這張圖的框；點列可選取對應的框。 */
function renderLabelTable() {
  const body = $('lblTableBody');
  if (!S.cur || !S.cur.boxes.length) {
    body.innerHTML = `<tr class="empty"><td colspan="4">${S.cur ? '尚未標註' : '尚未開啟資料集'}</td></tr>`;
    $('lblCount').textContent = S.cur ? '0 框' : '–';
    return;
  }
  $('lblCount').textContent = `${S.cur.boxes.length} 框`;
  body.innerHTML = S.cur.boxes.map((b, i) => {
    const d = boxDims640(b), pct = areaPct(b), small = isTooSmall(b);
    const cls = [i === S.sel ? 'sel' : '', small ? 'small' : '', b.source === 'ai' ? 'ai' : ''].filter(Boolean).join(' ');
    return `<tr data-i="${i}" class="${cls}">
      <td>${i + 1}</td>
      <td>${b.source === 'ai' ? '<i class="ai-mark">🤖</i>' : `<i class="dot" style="background:${colorOf(b.className)}"></i>`}${b.className}${b.readonly ? ' (唯讀)' : ''}</td>
      <td>${Math.round(d.w)} x ${Math.round(d.h)}${small ? ' ⚠' : ''}</td>
      <td>${pct.toFixed(1)}%</td>
    </tr>`;
  }).join('');
  body.querySelectorAll('tr[data-i]').forEach(tr => {
    tr.onclick = () => { S.sel = +tr.dataset.i; draw(); };
  });
}

function fit() {
  const stage = $('stage');
  const maxW = stage.clientWidth, maxH = stage.clientHeight;
  if (!S.cur || !maxW || !maxH) return;
  const s = Math.min(maxW / S.cur.width, maxH / S.cur.height, 1);
  view.scale = s || 1;
  cv.width = Math.round(S.cur.width * view.scale);
  cv.height = Math.round(S.cur.height * view.scale);
}
function draw() {
  if (!S.cur) return;
  fit();
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (img.complete && img.naturalWidth) ctx.drawImage(img, 0, 0, cv.width, cv.height);
  S.cur.boxes.forEach((b, i) => {
    const c = colorOf(b.className);
    const small = isTooSmall(b), ai = b.source === 'ai';
    const x = b.x * view.scale, y = b.y * view.scale, w = b.w * view.scale, h = b.h * view.scale;
    // 顏色優先序：唯讀灰 > 太小紅虛線 > AI 預標藍虛線 > 類別色
    const stroke = b.readonly ? '#94a3b8' : small ? '#dc2626' : ai ? AI_COLOR : c;
    ctx.lineWidth = i === S.sel ? 3 : 2;
    ctx.strokeStyle = stroke;
    if (small) ctx.setLineDash([4, 3]);
    else if (ai) ctx.setLineDash([8, 4]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    ctx.fillStyle = stroke;
    const label = (ai ? '🤖 ' : '') + b.className + (ai && b.conf != null ? ` ${Math.round(b.conf * 100)}%` : '') +
      (b.readonly ? ' (唯讀)' : small ? ' ⚠太小' : '');
    ctx.font = '600 12px system-ui';
    const tw = ctx.measureText(label).width + 10;
    ctx.fillRect(x, Math.max(0, y - 17), tw, 17);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x + 5, Math.max(12, y - 5));
    // 選取中且可編輯 → 畫出 8 個控制點，拖曳即可改大小
    if (i === S.sel && !b.readonly) {
      ctx.fillStyle = '#fff'; ctx.strokeStyle = c; ctx.lineWidth = 2;
      handlePoints(x, y, w, h).forEach(p => {
        ctx.beginPath();
        ctx.rect(p.x - HANDLE / 2, p.y - HANDLE / 2, HANDLE, HANDLE);
        ctx.fill(); ctx.stroke();
      });
    }
  });
  if (drag && drag.mode === 'new') {
    ctx.setLineDash([5, 4]); ctx.strokeStyle = PALETTE[S.classIdx % PALETTE.length]; ctx.lineWidth = 2;
    ctx.strokeRect(drag.x0, drag.y0, drag.x1 - drag.x0, drag.y1 - drag.y0);
    ctx.setLineDash([]);
  }
  // 有選取框時，讓類別下拉選單顯示這個框目前的類別，直接在選單裡選別的即可改類別
  if (S.sel >= 0 && S.cur.boxes[S.sel]) {
    const ci = S.ds ? S.ds.classes.indexOf(S.cur.boxes[S.sel].className) : -1;
    if (ci >= 0) $('classSel').value = String(ci);
  }
  renderLabelTable();
}
window.addEventListener('resize', () => { if (S.page === 'label') draw(); });

const HANDLE = 9;          // 控制點邊長（畫布像素）
const HANDLE_HIT = 7;      // 命中容差

/** 八個控制點（畫布座標）。貼齊圖片邊緣的框，控制點會有一半落在畫布外面而抓不到，
    所以把控制點往內夾——畫出來看得到，也點得到。 */
function handlePoints(x, y, w, h) {
  const m = HANDLE / 2;
  const cx = v => Math.max(m, Math.min(v, cv.width - m));
  const cy = v => Math.max(m, Math.min(v, cv.height - m));
  return [
    { k: 'nw', x: cx(x),         y: cy(y) },
    { k: 'n',  x: cx(x + w / 2), y: cy(y) },
    { k: 'ne', x: cx(x + w),     y: cy(y) },
    { k: 'e',  x: cx(x + w),     y: cy(y + h / 2) },
    { k: 'se', x: cx(x + w),     y: cy(y + h) },
    { k: 's',  x: cx(x + w / 2), y: cy(y + h) },
    { k: 'sw', x: cx(x),         y: cy(y + h) },
    { k: 'w',  x: cx(x),         y: cy(y + h / 2) },
  ];
}

const CURSORS = { nw: 'nwse-resize', se: 'nwse-resize', ne: 'nesw-resize', sw: 'nesw-resize',
                  n: 'ns-resize', s: 'ns-resize', e: 'ew-resize', w: 'ew-resize' };

/** 游標是否落在選取框的某個控制點上 */
function hitHandle(p) {
  if (S.sel < 0 || !S.cur) return null;
  const b = S.cur.boxes[S.sel];
  if (!b || b.readonly) return null;
  const pts = handlePoints(b.x * view.scale, b.y * view.scale, b.w * view.scale, b.h * view.scale);
  return pts.find(q => Math.abs(p.x - q.x) <= HANDLE_HIT && Math.abs(p.y - q.y) <= HANDLE_HIT) || null;
}

/** 由拖曳中的控制點算出新的框（影像座標），並夾在圖片範圍內 */
function resizeBox(orig, handle, dx, dy) {
  let { x, y, w, h } = orig;
  let x0 = x, y0 = y, x1 = x + w, y1 = y + h;
  if (handle.includes('w')) x0 = x + dx;
  if (handle.includes('e')) x1 = x + w + dx;
  if (handle.includes('n')) y0 = y + dy;
  if (handle.includes('s')) y1 = y + h + dy;
  if (x1 < x0) [x0, x1] = [x1, x0];        // 拖過頭翻面，維持正的寬高
  if (y1 < y0) [y0, y1] = [y1, y0];
  x0 = Math.max(0, Math.min(x0, S.cur.width));
  x1 = Math.max(0, Math.min(x1, S.cur.width));
  y0 = Math.max(0, Math.min(y0, S.cur.height));
  y1 = Math.max(0, Math.min(y1, S.cur.height));
  return { x: x0, y: y0, w: Math.max(2, x1 - x0), h: Math.max(2, y1 - y0) };
}

const pos = (e) => {
  const r = cv.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
};
cv.addEventListener('contextmenu', e => e.preventDefault());   // 右鍵挪去做選取，不要跳出瀏覽器選單

cv.addEventListener('pointerdown', e => {
  if (!S.cur) return;
  const p = pos(e);

  if (e.button === 2) {           // 右鍵：只負責選取（由上層往下找，蓋在上面的優先），不拖曳
    const idx = [...S.cur.boxes].reverse().findIndex(b =>
      p.x >= b.x * view.scale && p.x <= (b.x + b.w) * view.scale &&
      p.y >= b.y * view.scale && p.y <= (b.y + b.h) * view.scale);
    S.sel = idx >= 0 ? S.cur.boxes.length - 1 - idx : -1;
    draw();
    return;
  }
  if (e.button !== 0) return;     // 只處理左鍵／觸控

  // ① 目前選取的框：抓到它的控制點 → 改大小
  const hh = hitHandle(p);
  if (hh) {
    const b = S.cur.boxes[S.sel];
    pushHist();
    drag = { mode: 'resize', handle: hh.k, sx: p.x, sy: p.y, orig: { x: b.x, y: b.y, w: b.w, h: b.h } };
    cv.setPointerCapture(e.pointerId);
    return;
  }

  // ② 目前選取的框：點在它自己身上 → 拖曳移動。只認「目前選取的這一個」，不吃任何其他框的點擊，
  //    這樣才能在別的框（甚至互相重疊的框）上面重新拉一個新框，不會被壓在下面/上面的框搶走這次點擊。
  const sb = S.sel >= 0 ? S.cur.boxes[S.sel] : null;
  if (sb && !sb.readonly &&
      p.x >= sb.x * view.scale && p.x <= (sb.x + sb.w) * view.scale &&
      p.y >= sb.y * view.scale && p.y <= (sb.y + sb.h) * view.scale) {
    pushHist();
    drag = { mode: 'move', sx: p.x, sy: p.y, orig: { x: sb.x, y: sb.y, w: sb.w, h: sb.h } };
    cv.setPointerCapture(e.pointerId);
    return;
  }

  // ③ 其他地方（空白處，或別的框上面）→ 結束目前的編輯狀態，拉出新框
  S.sel = -1;
  drag = { mode: 'new', x0: p.x, y0: p.y, x1: p.x, y1: p.y };
  cv.setPointerCapture(e.pointerId);
  draw();
});

cv.addEventListener('pointermove', e => {
  const p = pos(e);
  if (!drag) {                                   // 沒在拖 → 只更新游標形狀
    const hh = hitHandle(p);
    cv.style.cursor = hh ? CURSORS[hh.k] : 'crosshair';
    return;
  }
  if (drag.mode === 'new') { drag.x1 = p.x; drag.y1 = p.y; draw(); return; }
  const dx = (p.x - drag.sx) / view.scale, dy = (p.y - drag.sy) / view.scale;
  const b = S.cur.boxes[S.sel];
  if (!b) return;
  if (drag.mode === 'move') {
    b.x = Math.max(0, Math.min(drag.orig.x + dx, S.cur.width - drag.orig.w));
    b.y = Math.max(0, Math.min(drag.orig.y + dy, S.cur.height - drag.orig.h));
  } else if (drag.mode === 'resize') {
    Object.assign(b, resizeBox(drag.orig, drag.handle, dx, dy));
  }
  draw();
});

cv.addEventListener('pointerup', () => {
  if (!drag) return;
  const mode = drag.mode;
  if (mode === 'new') {
    const x = Math.min(drag.x0, drag.x1) / view.scale, y = Math.min(drag.y0, drag.y1) / view.scale;
    const w = Math.abs(drag.x1 - drag.x0) / view.scale, h = Math.abs(drag.y1 - drag.y0) / view.scale;
    drag = null;
    if (w < 5 || h < 5) return draw();
    if (!S.ds.classes.length) { toast('請先在資料集頁新增類別', 'err'); return draw(); }
    S.classIdx = +$('classSel').value || 0;
    pushHist();
    S.cur.boxes.push({ className: S.ds.classes[S.classIdx], x, y, w, h });
    S.sel = S.cur.boxes.length - 1;
    syncDirty(); draw();
    if (isTooSmall({ w, h })) {
      const d = boxDims640({ w, h }), pct = areaPct({ w, h });
      toast(`這個框換算 640×640 輸入後只有 ${Math.round(d.w)}×${Math.round(d.h)} px（面積 ${pct.toFixed(1)}%），可能太小，建議放大或刪除`, 'err');
    }
    return;
  }
  const b = S.cur.boxes[S.sel], o = drag.orig;
  drag = null;
  if (b && o && (b.x === o.x && b.y === o.y && b.w === o.w && b.h === o.h)) S.hist.pop();  // 沒動到就不留歷程
  else if (b) markReviewed(b);
  syncDirty();
  draw();
});
document.addEventListener('keydown', e => {
  if (S.page !== 'label' || !$('cfModal').hidden || !$('fsModal').hidden) return;
  if ((e.key === 'Delete' || e.key === 'Backspace') && S.sel >= 0) {
    e.preventDefault(); pushHist(); S.cur.boxes.splice(S.sel, 1); S.sel = -1; syncDirty(); draw();
  } else if (e.key === 'ArrowRight') { $('nextImg').click(); }
  else if (e.key === 'ArrowLeft') { $('prevImg').click(); }
  else if (/^[1-9]$/.test(e.key)) { setClass(+e.key - 1); }
});
$('btnClear').onclick = () => { if (S.cur) { pushHist(); S.cur.boxes = S.cur.boxes.filter(b => b.readonly); syncDirty(); draw(); } };
$('btnBg').onclick = () => { if (S.cur) { pushHist(); S.cur.boxes = []; syncDirty(); draw(); } };

/* ---- 移除 / 復原單張圖片 ---- */
async function refreshRemovedState() {
  try {
    const r = await api('/dataset/image/removed');
    const b = $('btnRestoreImg');
    b.hidden = !r.has;
    if (r.has) {
      S.lastBatchCount = r.last_batch_count || 1;
      b.textContent = S.lastBatchCount > 1 ? `復原整批 (${S.lastBatchCount} 張)` : `復原移除的圖片 (${r.count})`;
    }
  } catch (e) { $('btnRestoreImg').hidden = true; }
}

async function refreshTestsetState() {
  try {
    const t = await api('/dataset/testset');
    $('btnUnTest').hidden = !t.last_batch_count;
    $('btnUnTest').textContent = `移回訓練集 (${t.last_batch_count} 張)`;
  } catch (e) { $('btnUnTest').hidden = true; }
}

$('btnUnTest').onclick = async () => {
  try {
    const r = await post('/dataset/testset/restore', {});
    renderDataset(r.summary);
    await loadPage(S.list.offset);
    refreshTestsetState();
    toast(`已移回 ${r.restored} 張到訓練資料` + (r.failed.length ? `；${r.failed.length} 張未移回：${r.failed.join('、')}` : ''),
      r.failed.length ? 'err' : 'ok');
  } catch (e) { toast('移回失敗：' + e.message, 'err'); }
};

/** 移除一或多張圖片之後共用的收尾：重新整理清單，盡量停在附近的一張圖。 */
async function afterImagesRemoved(keepIdx) {
  setDirty(false); S.hist = []; S.cur = null;
  renderDataset(await api('/dataset/summary'));
  await loadPage(S.list.offset);
  if (S.list.items.length) {
    selectImage(Math.min(keepIdx, S.list.items.length - 1));
  } else if (S.list.offset > 0) {
    await loadPage(Math.max(0, S.list.offset - S.list.limit));
    if (S.list.items.length) selectImage(S.list.items.length - 1);
  } else {
    $('imgName').textContent = '（依目前篩選已無圖片）';
    $('imgPos').textContent = '–'; cv.width = 0; cv.height = 0;
  }
  refreshRemovedState();
  refreshTestsetState();
}

async function removeImage() {
  if (!S.cur) return;
  const ok = await confirmDialog(
    '移除這張圖片？\n\n' + S.cur.path +
    '\n\n圖片與它的標籤檔會搬到 DefectMaster_Output/.removed_images/，不會出現在訓練資料裡，可按「復原移除的圖片」還原。' +
    unsavedNote(),
    { title: '移除此圖', okText: '移除' });
  if (!ok) return;
  const keepIdx = S.curIdx;
  try {
    await post('/dataset/image/remove', { path: S.cur.path });
    await afterImagesRemoved(keepIdx);
    toast('已移除，可按「復原移除的圖片」還原', 'ok');
  } catch (e) { toast('移除失敗：' + e.message, 'err'); }
}

async function restoreImage() {
  try {
    const url = S.lastBatchCount > 1 ? '/dataset/image/restore_batch' : '/dataset/image/restore';
    const r = await post(url, {});
    renderDataset(await api('/dataset/summary'));
    await loadPage(S.list.offset);
    refreshRemovedState();
    const n = r.restored != null ? r.restored : 1;
    const failMsg = r.failed && r.failed.length ? `，${r.failed.length} 張失敗（原位置已有同名檔案）` : '';
    toast(`已復原 ${n} 張${failMsg}` + (r.name ? `：${r.name}` : ''), 'ok');
  } catch (e) { toast('復原失敗：' + e.message, 'err'); }
}
$('btnRemoveImg').onclick = removeImage;
$('btnRestoreImg').onclick = restoreImage;

$('btnSave').onclick = saveCurrent;
$('btnUndo').onclick = undoStep;
document.addEventListener('keydown', e => {
  if (S.page !== 'label' || !(e.ctrlKey || e.metaKey) || !$('cfModal').hidden || !$('fsModal').hidden) return;
  const k = e.key.toLowerCase();
  if (k === 's') { e.preventDefault(); saveCurrent(); }
  if (k === 'z') { e.preventDefault(); undoStep(); }
});

/** 退一步：優先退本機還沒存檔的編輯（不跳視窗），本機沒得退了才退磁碟上的版本。 */
async function undoStep() {
  if (!S.cur) return;
  if (S.hist.length) {
    S.cur.boxes = JSON.parse(S.hist.pop());
    S.sel = -1;
    syncDirty();
    draw();
    $('saveState').textContent = S.dirty ? '尚未儲存' : '已回到磁碟上的內容';
    return;
  }
  await undoCurrent();      // 本機歷程空了 → 退已存檔的版本（會先確認）
}
window.addEventListener('beforeunload', e => { if (S.dirty) { e.preventDefault(); e.returnValue = ''; } });

async function saveCurrent() {
  if (!S.cur || !S.dirty) return;
  const tiny = S.cur.boxes
    .map((b, i) => ({ i, b, d: boxDims640(b), pct: areaPct(b) }))
    .filter(x => isTooSmall(x.b));
  if (tiny.length) {
    const lines = tiny.map(x =>
      `· #${x.i + 1} ${x.b.className}：${Math.round(x.d.w)}×${Math.round(x.d.h)} px（面積 ${x.pct.toFixed(1)}%）`
    ).join('\n');
    const proceed = await confirmDialog(
      `有 ${tiny.length} 個框換算 640×640 輸入後寬或高小於 ${MIN_BOX_DIM_PX}px，容易讓模型學到雜訊：\n\n` +
      lines + '\n\n建議先放大或刪除這些框（右側清單也能點選定位），再儲存。',
      { title: '框太小', okText: '仍要儲存', cancelText: '先調整' });
    if (!proceed) return;
  }
  const n = S.cur.boxes.filter(b => !b.readonly).length;
  const aiN = S.cur.boxes.filter(b => b.source === 'ai').length;
  const ok = await confirmDialog(
    '即將覆寫原始標籤檔：\n' + labelPathHint(S.cur.path) +
    `\n\n這張圖會存成 ${n} 個標註框` + (n === 0 ? '（空檔＝無瑕疵背景圖）' : '') +
    (aiN ? `，其中 ${aiN} 個是 AI 預標、還沒手動調整過` : '') +
    '\n\n原本的標籤檔會先自動備份，存檔後仍可按「復原」還原。',
    { title: '儲存標註', okText: '寫入', dontAskKey: 'save_label' });
  if (!ok) return;
  try {
    const r = await post('/dataset/labels', { path: S.cur.path, boxes: S.cur.boxes });
    S.cur.boxes.forEach(markReviewed);           // 存進 txt 之後就是一般標註了，🤖 標記一併拿掉
    S.baseline = JSON.stringify(S.cur.boxes);   // 存檔後這就是新的磁碟內容
    S.hist = [];
    setDirty(false);
    $('saveState').textContent = '已存檔 ' + new Date().toLocaleTimeString();
    draw();
    const t = document.querySelector(`.thumb[data-i="${S.curIdx}"]`);
    if (t && !t.querySelector('i') && S.cur.boxes.length) t.insertAdjacentHTML('beforeend', '<i></i>');
    if (S.list.items[S.curIdx]) S.list.items[S.curIdx].labeled = S.cur.boxes.length > 0;
    await refreshBackupState();
    renderDataset(await api('/dataset/summary'));
    toast('已寫入 ' + r.label_path, 'ok');
  } catch (e) { toast('存檔失敗：' + e.message, 'err'); }
}

async function undoCurrent() {
  if (!S.cur) return;
  // 有未存檔的修改 → 先丟掉，回到磁碟上的內容（檔案不動）
  if (S.dirty) {
    if (!await confirmDialog('要放棄這張圖尚未儲存的修改，回到磁碟上的標籤嗎？', { title: '放棄修改', okText: '放棄' })) return;
    const d = await api('/dataset/labels?path=' + encodeURIComponent(S.cur.path));
    S.cur.boxes = d.boxes; S.sel = -1; S.hist = [];
    S.baseline = JSON.stringify(d.boxes);
    setDirty(false); draw();
    toast('已放棄未儲存的修改');
    return;
  }
  if (!S.hasBackup) return;
  if (!await confirmDialog(`要退回上一個已儲存的版本嗎？\n\n目前這張圖還有 ${S.undoCount} 層還原點，可以連續往回退。`, { title: '復原', okText: '退回' })) return;
  try {
    const r = await post('/dataset/labels/restore', { path: S.cur.path });
    S.cur.boxes = r.boxes;
    S.baseline = JSON.stringify(r.boxes);
    S.hist = [];
    setDirty(false);
    draw();
    $('saveState').textContent = '已還原至 ' + (r.restored_to || '存檔前');
    const t = document.querySelector(`.thumb[data-i="${S.curIdx}"]`);
    if (t) {
      const dot = t.querySelector('i');
      if (r.boxes.length && !dot) t.insertAdjacentHTML('beforeend', '<i></i>');
      if (!r.boxes.length && dot) dot.remove();
    }
    if (S.list.items[S.curIdx]) S.list.items[S.curIdx].labeled = r.boxes.length > 0;
    S.sel = -1;
    await refreshBackupState();
    renderDataset(await api('/dataset/summary'));
    toast(r.remaining > 0 ? `已退回上一版，還有 ${r.remaining} 層可退` : '已退回到最初的標籤', 'ok');
  } catch (e) { toast('還原失敗：' + e.message, 'err'); }
}

/** 依 Ultralytics 慣例推出標籤檔位置，只是給確認視窗顯示用。 */
function labelPathHint(imgPath) {
  const i = imgPath.lastIndexOf('/');
  let dir = imgPath.slice(0, i), name = imgPath.slice(i + 1);
  const k = dir.lastIndexOf('/images');
  if (k >= 0) dir = dir.slice(0, k) + '/labels' + dir.slice(k + 7);
  return dir + '/' + name.replace(/\.[^.]+$/, '') + '.txt';
}

/* ---------------- 訓練 ---------------- */
async function loadWeights() {
  const [w, m] = await Promise.all([api('/weights'), api('/models')]);
  const sel = $('trModel');
  const keep = sel.value;
  sel.innerHTML = `<optgroup label="官方權重">${w.weights.map(x => `<option>${x}</option>`).join('')}</optgroup>` +
                  (m.models.length ? `<optgroup label="我的模型（Fine-tune）">${m.models.map(x => `<option>${x}</option>`).join('')}</optgroup>` : '');
  sel.value = keep || (w.weights.includes('yolo11n.pt') ? 'yolo11n.pt' : (w.weights[0] || ''));
}
$('btnTrain').onclick = async () => {
  if (!S.ds || !S.ds.open) return toast('請先載入資料集', 'err');
  const aug = $('trAug').checked ? { hsv_h: 0.015, hsv_s: 0.5, hsv_v: 0.35, fliplr: 0.5, scale: 0.4, translate: 0.1, mosaic: 1.0 }
                                 : { hsv_h: 0, hsv_s: 0, hsv_v: 0, fliplr: 0, scale: 0, translate: 0, mosaic: 0 };
  try {
    const t = await api('/infer/add_target');
    if (t.pending && !await confirmDialog(
      `有 ${t.pending} 張從推論頁加入的圖片還沒在「標註工具」存檔確認過。\n\n` +
      '它們目前帶的是模型自己的預測框（沒有框的會被當成背景圖），直接訓練等於讓模型學自己的錯誤。\n\n' +
      '建議先到標註工具（篩選「從推論加入」）修正並存檔。',
      { title: '有尚未確認的預標註', okText: '仍要開始訓練' })) return;
  } catch (e) { /* 檢查失敗不擋訓練 */ }
  try {
    const r = await post('/train', {
      model: $('trModel').value, epochs: +$('trEpochs').value, batch: +$('trBatch').value,
      imgsz: +$('trImgsz').value, workers: +$('trWorkers').value, name: $('trName').value, aug,
      tensorrt: $('trTrt').checked,
    });
    toast(`訓練已啟動：${r.name}`, 'ok');
    $('btnTrain').disabled = true; $('btnStop').disabled = false;
    pollTrain();
  } catch (e) { toast(e.message, 'err'); }
};
$('btnStop').onclick = async () => { await post('/train/stop'); toast('已送出停止訊號'); };

function pollTrain() {
  clearInterval(S.polling);
  S.polling = setInterval(async () => {
    let s;
    try { s = await api('/train/status'); } catch (e) { return; }
    const b = $('trState');
    b.textContent = s.phase === 'export' ? 'TensorRT 轉換中' : s.state;
    b.className = 'badge ' + ({ running: 'run', done: 'ok', error: 'err', export_failed: 'err' }[s.state] || 'muted');
    $('progBar').style.width = s.epochs ? (100 * s.epoch / s.epochs).toFixed(1) + '%' : '0';
    $('trStats').innerHTML = stat('Epoch', `${s.epoch}/${s.epochs}`) +
      stat('已耗時', fmtTime(s.elapsed)) +
      Object.entries(s.metrics || {}).filter(([k]) => /mAP|precision|recall/i.test(k))
        .map(([k, v]) => stat(k, v)).join('');
    const el = $('trLogs');
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 40;
    el.textContent = (s.logs || []).join('\n') || '尚未開始。';
    if (atBottom) el.scrollTop = el.scrollHeight;
    if (s.state !== 'running') {
      clearInterval(S.polling); S.polling = null;
      $('btnTrain').disabled = false; $('btnStop').disabled = true;
      if (s.state === 'done') {
        toast(s.engine ? ' 訓練與 TensorRT 轉換都完成了' : '訓練完成，記憶體已釋放', 'ok');
        loadRuns();
        loadModels().catch(() => {});      // 新模型要立刻出現在推論頁，且排在最上面
      }
      if (s.state === 'export_failed')
        toast('訓練成功，但 TensorRT 轉換失敗：' + (s.error || '見下方 log'), 'err');
      if (s.state === 'error') toast('訓練失敗：' + (s.error || ''), 'err');
    }
  }, 2000);
}
const fmtTime = (s) => `${String((s / 60) | 0).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

/* ---------------- 評估 ---------------- */
async function loadRuns() {
  const r = await api('/runs');
  const sel = $('runSel');
  const keep = sel.value;
  sel.innerHTML = r.runs.map(x => `<option>${x.name}</option>`).join('') || '<option value="">（尚無訓練結果）</option>';
  sel.value = keep && r.runs.some(x => x.name === keep) ? keep : (r.runs[0] ? r.runs[0].name : '');
  showRun();
}
$('runSel').onchange = showRun;
$('btnRunReload').onclick = loadRuns;
async function showRun() {
  const name = $('runSel').value;
  if (!name) { $('evalStats').innerHTML = ''; $('evalPlots').innerHTML = ''; return; }
  const r = await api('/run?name=' + encodeURIComponent(name));
  const m = r.metrics || {};
  $('evalStats').innerHTML = ['mAP', 'mAP50', 'Precision', 'Recall', 'F1', 'Accuracy_Top1']
    .filter(k => m[k] !== undefined).map(k => stat(k, m[k])).join('') +
    (m.base_model ? stat('base', m.base_model) : '');
  $('evalPlots').innerHTML = r.plots.map(p =>
    `<figure><img loading="lazy" src="/image?path=${encodeURIComponent(r.dir + '/' + p)}"><figcaption>${p}</figcaption></figure>`).join('')
    || '<p class="hint">此結果沒有圖表檔。</p>';
}

/** 只有目前資料集套用了有 test 集的切分，才顯示「在測試集上評估」卡片。 */
async function loadTestEvalCard() {
  let d;
  try { d = await api('/dataset/summary'); } catch (e) { $('testEvalCard').hidden = true; return; }
  const nSplit = d.open && d.split && d.split.counts ? d.split.counts.test || 0 : 0;
  const nFolder = d.open && d.testset ? d.testset.labeled : 0;
  $('testEvalCard').hidden = !(nSplit || nFolder);
  if (!(nSplit || nFolder)) return;
  const src = $('testEvalSrc'), keepSrc = src.value;
  src.innerHTML = (nFolder ? `<option value="folder">測試圖片資料夾（${nFolder} 張已標註）</option>` : '') +
    (nSplit ? `<option value="split">切分的 test 集（${nSplit} 張）</option>` : '');
  src.value = [...src.options].some(o => o.value === keepSrc) ? keepSrc : src.options[0].value;
  const [m, w] = await Promise.all([api('/models'), api('/weights')]);
  const sel = $('testEvalModel');
  const keep = sel.value;
  sel.innerHTML = `<optgroup label="我的模型">${m.models.map(x => `<option>${x}</option>`).join('')}</optgroup>` +
    `<optgroup label="官方權重">${w.weights.map(x => `<option>${x}</option>`).join('')}</optgroup>`;
  sel.value = keep || m.models[0] || w.weights[0] || '';
}
$('btnTestEval').onclick = async () => {
  const model = $('testEvalModel').value;
  if (!model) return toast('沒有可用的模型', 'err');
  $('btnTestEval').disabled = true;
  $('testEvalStats').innerHTML = '<span class="hint">評估中，可能需要幾十秒到幾分鐘…</span>';
  try {
    const folder = $('testEvalSrc').value === 'folder';
    const r = await post(folder ? '/dataset/testset/evaluate' : '/dataset/split/evaluate', { model });
    $('testEvalStats').innerHTML = Object.entries(r.metrics || {}).map(([k, v]) => stat(k, v)).join('') +
      (r.images != null ? stat('影像數', r.images) : '') + stat('模型', model) +
      stat('來源', folder ? '測試圖片資料夾' : '切分 test 集');
    toast('測試集評估完成', 'ok');
  } catch (e) {
    $('testEvalStats').innerHTML = '';
    toast('評估失敗：' + e.message, 'err');
  } finally { $('btnTestEval').disabled = false; }
};

/* ---------------- 推論 ---------------- */
async function loadModels() {
  const [m, w] = await Promise.all([api('/models'), api('/weights')]);
  const sel = $('ifModel');
  const keep = sel.value;
  const engines = m.engines || [];
  sel.innerHTML =
    (engines.length ? `<optgroup label="TensorRT engine（FP16，Jetson 最快）">${engines.map(e =>
      `<option value="${e.name}">${e.name}${e.imgsz ? ` — imgsz ${e.imgsz}` : ''}${e.half === false ? ' FP32' : ' FP16'}</option>`
    ).join('')}</optgroup>` : '') +
    `<optgroup label="我的模型（.pt）">${m.models.map(x => `<option>${x}</option>`).join('')}</optgroup>` +
    `<optgroup label="官方權重">${w.weights.map(x => `<option>${x}</option>`).join('')}</optgroup>`;
  sel.value = keep || (engines[0] && engines[0].name) || m.models[0] || w.weights[0] || '';
  updateEngineUi();
}

/** .engine 的精度在建圖時就固定了，所以 FP16 勾選框只對 .pt 有意義。 */
function updateEngineUi() {
  const isEngine = $('ifModel').value.endsWith('.engine');
  $('ifEngineBadge').hidden = !isEngine;
  $('ifHalfWrap').style.opacity = isEngine ? '.45' : '1';
  $('ifHalf').disabled = isEngine;
  $('ifHalf').title = isEngine ? 'engine 的精度在建立時就決定了（FP16），不需要另外勾選' : '用半精度推論，速度較快';
}
$('ifModel').onchange = updateEngineUi;
$('ifConf').oninput = () => { $('ifConfVal').textContent = (+$('ifConf').value).toFixed(2); };
$('btnPickImg').onclick = () => pickPath('image',
  $('ifPath').value.trim() || (S.ds && S.ds.root) || '',
  '選擇要推論的圖片', p => { $('ifPath').value = p; runPredict(); });

/** 從目前開啟的資料集抓一張來測——剛訓練完想馬上看效果時最快。 */
$('btnPickFromDs').onclick = async () => {
  try {
    const s = await api('/dataset/summary');
    if (!s.open) return toast('尚未開啟資料集，請先到「資料集」頁載入，或用「瀏覽圖片」選檔', 'err');
    const off = Math.floor(Math.random() * s.total);
    const r = await api(`/dataset/images?offset=${off}&limit=1`);
    if (!r.items.length) return toast('資料集沒有可用的圖片', 'err');
    $('ifPath').value = r.items[0].path;
    await runPredict();
  } catch (e) { toast(e.message, 'err'); }
};
$('btnPredict').onclick = runPredict;
async function runPredict() {
  const path = $('ifPath').value.trim();
  if (!path) return toast('請輸入圖片路徑', 'err');
  $('btnPredict').disabled = true;
  try {
    const half = $('ifHalf').checked && !$('ifModel').value.endsWith('.engine');
    const r = await post('/predict', { model: $('ifModel').value, path, conf: +$('ifConf').value, half });
    if (!r.ok) throw new Error(r.error || '推論失敗');
    showPrediction(path, r, $('ifModel').value, half);
  } catch (e) { toast(e.message, 'err'); }
  finally { $('btnPredict').disabled = false; }
}

/** 把一筆推論結果畫進 ctx；s 是「原圖像素 → 畫布像素」的縮放比。 */
function drawDets(c, dets, s, fontPx = 12) {
  c.font = `600 ${fontPx}px system-ui`;
  dets.forEach(d => {
    c.strokeStyle = colorOf(d.cls); c.fillStyle = colorOf(d.cls); c.lineWidth = fontPx >= 12 ? 2 : 1.5;
    if (d.type === 'bbox') {
      c.strokeRect(d.x * s, d.y * s, d.w * s, d.h * s);
      const t = `${d.cls} ${(d.conf * 100).toFixed(0)}%`, th = fontPx + 5;
      c.fillRect(d.x * s, Math.max(0, d.y * s - th), c.measureText(t).width + 8, th);
      c.fillStyle = '#fff'; c.fillText(t, d.x * s + 4, Math.max(fontPx, d.y * s - 4));
    } else if (d.points) {
      c.beginPath();
      d.points.forEach((p, i) => i ? c.lineTo(p.x * s, p.y * s) : c.moveTo(p.x * s, p.y * s));
      c.closePath(); c.stroke();
    }
  });
}

/** 在上方大畫面顯示一筆推論結果（單張推論、或從批次結果點進來看都走這裡）。 */
function showPrediction(path, r, modelName, half) {
  S.infer = r;
  $('ifPath').value = path;
  const ic = $('ifCv'), ictx = ic.getContext('2d');
  const im = new Image();
  im.onload = () => {
    const stage = $('ifStage');
    const s = Math.min(stage.clientWidth / r.width, stage.clientHeight / r.height, 1);
    ic.width = r.width * s; ic.height = r.height * s;
    ictx.drawImage(im, 0, 0, ic.width, ic.height);
    drawDets(ictx, r.dets, s);
  };
  im.src = '/image?path=' + encodeURIComponent(path);
  $('ifStats').innerHTML = stat('模型', modelName.replace(/\.(pt|engine)$/, '')) +
    stat('格式', modelName.endsWith('.engine') ? 'TensorRT FP16' : (half ? 'PyTorch FP16' : 'PyTorch FP32')) +
    stat('偵測數', r.dets.length) +
    (r.speed && r.speed.inference ? stat('推論 ms', r.speed.inference) : '') +
    r.dets.slice(0, 6).map(d => stat(d.cls, (d.conf * 100).toFixed(0) + '%')).join('');
}
$('btnRelease').onclick = async () => { await post('/predict/release'); toast('推論模型已卸載，記憶體已釋放', 'ok'); refreshSys(); };

/* ---------------- 批次推論 ----------------
   一次推論整個資料夾。結果（只有框座標，不含圖）放在 BX.items，畫面一次只畫一頁縮圖。
   效果不好的圖按「加入訓練集」→ 複製進目前資料集的訓練資料夾並帶入預測框當預標註，
   之後在標註頁用「從推論加入」篩選出來修正。 */
const BX = { items: [], running: false, stop: false, page: 0, per: 48, done: 0, model: '', conf: 0.25, half: false, cur: -1 };
const esc = (v) => String(v).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const LOW_CONF = 0.5;

const bxMinConf = (it) => (it.r && it.r.dets.length) ? Math.min(...it.r.dets.map(d => d.conf)) : 0;

$('btnBxPick').onclick = () => pickPath('folder',
  $('bxPath').value.trim() || (S.ds && S.ds.root) || '',
  '選擇要批次推論的圖片資料夾', p => { $('bxPath').value = p; });
$('bxPath').addEventListener('keydown', e => { if (e.key === 'Enter' && !BX.running) bxRun(); });
$('btnBxRun').onclick = () => bxRun();
$('btnBxStop').onclick = () => { BX.stop = true; $('btnBxStop').disabled = true; };
$('bxFilter').onchange = () => { BX.page = 0; renderBx(); };
$('bxSort').onchange = () => { BX.page = 0; renderBx(); };
$('bxPrev').onclick = () => { if (BX.page > 0) { BX.page--; renderBx(); } };
$('bxNext').onclick = () => { if ((BX.page + 1) * BX.per < bxView().length) { BX.page++; renderBx(); } };

/** 目前篩選＋排序後要顯示的 BX.items 索引。 */
function bxView() {
  const f = $('bxFilter').value, so = $('bxSort').value;
  let idx = BX.items.map((_, i) => i).filter(i => {
    const it = BX.items[i];
    if (f === 'none') return it.r && !it.r.dets.length;
    if (f === 'low') return it.r && it.r.dets.some(d => d.conf < LOW_CONF);
    if (f === 'added') return !!it.added;
    if (f === 'error') return !!it.err;
    return true;
  });
  if (so === 'minconf') idx.sort((a, b) => (BX.items[a].r ? bxMinConf(BX.items[a]) : 2) - (BX.items[b].r ? bxMinConf(BX.items[b]) : 2));
  if (so === 'count') idx.sort((a, b) => (BX.items[a].r ? BX.items[a].r.dets.length : 1e9) - (BX.items[b].r ? BX.items[b].r.dets.length : 1e9));
  return idx;
}

function bxCardHtml(i) {
  const it = BX.items[i];
  let sum;
  if (it.err) sum = `<span class="bx-sum err" title="${esc(it.err)}">推論失敗</span>`;
  else if (!it.r) sum = '<span class="bx-sum">等待推論…</span>';
  else if (!it.r.dets.length) sum = '<span class="bx-sum none">沒有偵測到</span>';
  else {
    const cnt = {};
    it.r.dets.forEach(d => { cnt[d.cls] = (cnt[d.cls] || 0) + 1; });
    const detail = Object.entries(cnt).map(([k, v]) => `${k}×${v}`).join('  ');
    sum = `<span class="bx-sum" title="${esc(detail)}">${it.r.dets.length} 個框 · 最低 ${(bxMinConf(it) * 100).toFixed(0)}% · ${esc(detail)}</span>`;
  }
  const act = it.added
    ? `<button class="btn ghost sm" data-act="goto">已加入 · 去標註</button>` +
      (it.addedCopied !== false ? `<button class="btn ghost sm undo" data-act="undo" title="把加入的那份移除（可在標註頁復原）">撤銷</button>` : '')
    : `<button class="btn green sm" data-act="add" ${it.r || it.err ? '' : 'disabled'}>${it.inDs ? '已在資料集 · 去標註' : '加入訓練集'}</button>`;
  return `<div class="bx-card${it.added ? ' added' : ''}${BX.cur === i ? ' cur' : ''}" data-i="${i}">
    <div class="bx-img" data-act="view" title="點一下在上方大圖檢視">${it.r ? '<canvas></canvas>' : '<span class="bx-wait">…</span>'}</div>
    <div class="bx-meta"><span class="bx-name" title="${esc(it.path)}">${esc(it.name)}</span>${sum}</div>
    <div class="bx-act">${act}</div>
  </div>`;
}

function bxBindCard(el) {
  const i = +el.dataset.i, it = BX.items[i];
  el.querySelectorAll('[data-act]').forEach(b => {
    b.onclick = () => ({ view: bxShow, add: bxAdd, undo: bxUndo, goto: () => gotoLabel(it.added) })[b.dataset.act](i);
  });
  const c = el.querySelector('canvas');
  if (!c || !it.r) return;
  const im = new Image();
  im.onload = () => {
    c.width = im.naturalWidth; c.height = im.naturalHeight;
    const g = c.getContext('2d');
    g.drawImage(im, 0, 0);
    drawDets(g, it.r.dets, im.naturalWidth / it.r.width, 10);
  };
  im.src = `/image?w=360&path=${encodeURIComponent(it.path)}`;
}

function renderBx() {
  const view = bxView();
  const pages = Math.max(1, Math.ceil(view.length / BX.per));
  BX.page = Math.min(BX.page, pages - 1);
  const slice = view.slice(BX.page * BX.per, (BX.page + 1) * BX.per);
  $('bxPgInfo').textContent = view.length ? `${BX.page * BX.per + 1}-${BX.page * BX.per + slice.length} / ${view.length}` : '0 / 0';
  $('bxGrid').innerHTML = slice.map(bxCardHtml).join('') ||
    (BX.items.length ? '<p class="hint">這個篩選條件下沒有圖片。</p>' : '');
  $('bxGrid').querySelectorAll('.bx-card').forEach(bxBindCard);
  bxStats();
}

/** 只重畫一張卡（推論進行中每張做完就更新，不必整頁重來）。 */
function bxUpdateCard(i) {
  const old = $('bxGrid').querySelector(`.bx-card[data-i="${i}"]`);
  if (!old) return;
  const tmp = document.createElement('div');
  tmp.innerHTML = bxCardHtml(i);
  const el = tmp.firstElementChild;
  old.replaceWith(el);
  bxBindCard(el);
}

function bxStats() {
  if (!BX.items.length) { $('bxStats').innerHTML = ''; return; }
  const done = BX.items.filter(it => it.r);
  const ms = done.map(it => it.r.ms).filter(Boolean);
  $('bxStats').innerHTML = stat('圖片', BX.items.length) + stat('已推論', done.length) +
    stat('沒有偵測到', done.filter(it => !it.r.dets.length).length) +
    stat(`含低信心框 &lt;${LOW_CONF * 100}%`, done.filter(it => it.r.dets.some(d => d.conf < LOW_CONF)).length) +
    (ms.length ? stat('平均推論 ms', (ms.reduce((a, b) => a + b, 0) / ms.length).toFixed(1)) : '') +
    stat('已加入訓練集', BX.items.filter(it => it.added).length) +
    (BX.items.some(it => it.err) ? stat('失敗', BX.items.filter(it => it.err).length) : '');
}

async function refreshBxTarget() {
  const box = $('bxTarget');
  try {
    const t = await api('/infer/add_target');
    if (!t.open) {
      box.className = 'bx-target bad';
      box.textContent = '尚未開啟資料集：批次推論可以照常使用，但要「加入訓練集」請先到「資料集」頁載入要加入的訓練資料集。';
      return;
    }
    if (!t.dir) { box.className = 'bx-target bad'; box.textContent = '無法加入訓練集：' + t.error; return; }
    if (t.test_dir && !$('bxPath').value.trim()) $('bxPath').value = t.test_dir;
    box.className = 'bx-target';
    box.innerHTML = `「加入訓練集」會複製到：<b>${esc(t.dir)}</b>` +
      (t.mode === 'split' ? '（並補進目前切分的 train 清單）' : '') +
      `<br>預測框依類別名稱對應到資料集類別（${esc(t.classes.join(', '))}），對不上的類別會略過。` +
      (t.pending ? `<br><span style="color:var(--amber);font-weight:700">有 ${t.pending} 張從推論加入的圖片還沒在標註頁存檔確認。</span>
        <button class="btn ghost sm" id="bxGotoPending">到標註頁修正</button>` : '');
    if (t.pending) $('bxGotoPending').onclick = () => gotoLabel('');
  } catch (e) { box.className = 'bx-target bad'; box.textContent = e.message; }
}

async function bxRun() {
  const folder = $('bxPath').value.trim();
  if (!folder) return toast('請先選擇要批次推論的資料夾', 'err');
  const model = $('ifModel').value;
  if (!model) return toast('請先在上方選擇模型', 'err');
  let list;
  try { list = await api('/infer/folder?path=' + encodeURIComponent(folder)); }
  catch (e) { return toast(e.message, 'err'); }
  $('bxPath').value = list.folder;
  BX.items = list.items.map(x => ({ path: x.path, name: x.name, r: null, err: '',
                                    added: x.added || '', inDs: !!x.in_dataset }));
  BX.model = model; BX.conf = +$('ifConf').value;
  BX.half = $('ifHalf').checked && !model.endsWith('.engine');
  BX.running = true; BX.stop = false; BX.page = 0; BX.done = 0; BX.cur = -1;
  $('btnBxRun').disabled = true; $('btnBxStop').hidden = false; $('btnBxStop').disabled = false;
  $('bxToolbar').hidden = false; $('bxProgWrap').hidden = false; $('bxProg').style.width = '0%';
  $('bxBadge').hidden = false;
  renderBx();
  refreshBxTarget();

  let fails = 0;
  for (let i = 0; i < BX.items.length && !BX.stop; i++) {
    const it = BX.items[i];
    try {
      const r = await post('/predict', { model, path: it.path, conf: BX.conf, half: BX.half });
      if (!r.ok) throw new Error(r.error || '推論失敗');
      it.r = { dets: r.dets, width: r.width, height: r.height, speed: r.speed, ms: r.speed && r.speed.inference };
      fails = 0;
    } catch (e) {
      it.err = e.message;
      if (++fails >= 3 && !BX.items.some(x => x.r)) {      // 一開始就連續失敗 → 多半是模型本身有問題
        toast('連續推論失敗，已停止：' + e.message, 'err');
        BX.stop = true;
      }
    }
    BX.done = i + 1;
    $('bxProg').style.width = (100 * BX.done / BX.items.length).toFixed(1) + '%';
    $('bxBadge').textContent = `${BX.done} / ${BX.items.length} · ${model.replace(/\.(pt|engine)$/, '')} · conf ${BX.conf.toFixed(2)}`;
    bxUpdateCard(i);
    if (i % 5 === 4) bxStats();
  }
  BX.running = false;
  $('btnBxRun').disabled = false; $('btnBxStop').hidden = true;
  renderBx();
  if (BX.stop && BX.done < BX.items.length) toast(`已停止，完成 ${BX.done} / ${BX.items.length} 張`);
  else toast(`批次推論完成：${BX.items.length} 張`, 'ok');
}

function bxShow(i) {
  const it = BX.items[i];
  if (!it.r) return toast(it.err || '這張還沒推論', it.err ? 'err' : undefined);
  const prev = BX.cur;
  BX.cur = i;
  if (prev >= 0) bxUpdateCard(prev);
  bxUpdateCard(i);
  showPrediction(it.path, it.r, BX.model, BX.half);
  $('ifStage').scrollIntoView({ behavior: 'smooth', block: 'center' });
}

async function bxAdd(i) {
  const it = BX.items[i];
  if (!S.ds || !S.ds.open) return toast('尚未開啟資料集，請先到「資料集」頁載入要加入的訓練資料集', 'err');
  try {
    const r = await post('/infer/add', { path: it.path, dets: it.r ? it.r.dets : [], model: BX.model });
    it.added = r.dest;
    it.addedCopied = r.status !== 'exists' || r.dest !== it.path;
    renderDataset(r.summary);
    resetLabelListIfStale();
    if (r.status === 'exists') toast('這張已經在資料集裡了，按「去標註」直接修正', 'ok');
    else toast(`已加入 ${r.dest.split('/').pop()}，帶入 ${r.boxes} 個預標註框` +
      (r.skipped_classes.length ? `；類別 ${r.skipped_classes.join(', ')} 不在資料集中，已略過` : '') +
      (r.boxes ? '' : '（沒有可用的框，在標註頁會顯示為未標註）'), 'ok');
    bxUpdateCard(i); bxStats(); refreshBxTarget();
  } catch (e) { toast('加入失敗：' + e.message, 'err'); }
}

async function bxUndo(i) {
  const it = BX.items[i];
  try {
    const r = await post('/infer/add/undo', { dest: it.added });
    it.added = ''; it.inDs = false;
    renderDataset(r.summary);
    resetLabelListIfStale();
    toast('已撤銷加入（檔案移到 .removed_images/，標註頁可復原）', 'ok');
    bxUpdateCard(i); bxStats(); refreshBxTarget();
  } catch (e) { toast('撤銷失敗：' + e.message, 'err'); }
}

/** 資料集內容被推論頁改過 → 標註頁的清單下次進去要重新拿。 */
function resetLabelListIfStale() { if (S.page !== 'label') S.list.items = []; }

/** 切到標註頁並篩選「從推論加入」；path 有值就直接打開那張圖。 */
function gotoLabel(path) {
  S.list.filter = 'infer_added';
  $('filterSel').value = 'infer_added';
  S.labelFocus = { path: path || '' };
  document.querySelector('.nav-item[data-page="label"]').click();
}

/* ---------------- 系統狀態 ---------------- */
async function refreshSys() {
  try {
    const s = await api('/sysinfo');
    $('memServer').textContent = s.server_rss_mb + ' MB';
    $('memWorker').textContent = s.workers.length
      ? s.workers.map(w => `${w.kind} ${w.rss}MB`).join(' · ') : '無';
    $('memFree').textContent = `${s.mem_available_mb} / ${s.mem_total_mb} MB`;
    const used = s.mem_total_mb ? 100 * (1 - s.mem_available_mb / s.mem_total_mb) : 0;
    $('memBar').style.width = used.toFixed(0) + '%';
  } catch (e) { /* server 尚未就緒 */ }
}
async function boot() {
  try {
    const l = await api('/license/status');
    const box = $('licBox');
    box.className = 'lic ' + (l.licensed ? 'ok' : 'bad');
    box.textContent = l.licensed ? `授權有效 · ${l.mac}` : l.message;
  } catch (e) { $('licBox').textContent = '無法連線到 Server'; }
  try {
    const d = await api('/defaults');
    S.picker = d.file_picker || 'native';
    if (d.default_yaml && !$('yamlPath').value) $('yamlPath').value = d.default_yaml;
    if (d.default_folder && !$('folderPath').value) $('folderPath').value = d.default_folder;
  } catch (e) { /* 沒設定就算了 */ }
  try { renderDataset(await api('/dataset/summary')); } catch (e) { /* 尚未開啟 */ }
  loadWeights().catch(() => {});
  refreshSys();
  setInterval(refreshSys, 5000);
}
boot();
