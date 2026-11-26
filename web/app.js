'use strict';
const $ = (id) => document.getElementById(id);
const money = (value) => new Intl.NumberFormat('es-CL', {style: 'currency', currency: 'CLP', maximumFractionDigits: 0}).format(value);
let metadata;
let inputRevision = 0;
let compareRevision = 0;
let hasNeighbors = false;
window.chartsReady = false;

const navigationLinks = [...document.querySelectorAll('.site-header nav a')];
let navigationFrame;
function markCurrentSection() {
  const anchorOffset = parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop) || 0;
  const limit = Math.max(anchorOffset, document.querySelector('.header-shell').getBoundingClientRect().height) + 8;
  let current = navigationLinks[0];
  for (const link of navigationLinks) {
    const section = document.getElementById(link.hash.slice(1));
    if (section.getBoundingClientRect().top <= limit) current = link;
  }
  navigationLinks.forEach((link) => {
    link.classList.toggle('is-active', link === current);
    if (link === current) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
}
window.addEventListener('scroll', () => {
  cancelAnimationFrame(navigationFrame);
  navigationFrame = requestAnimationFrame(markCurrentSection);
}, {passive: true});
window.addEventListener('resize', markCurrentSection);
markCurrentSection();

async function request(url, options = {}) {
  const response = await fetch(url, {...options, signal: AbortSignal.timeout(20000)});
  const body = await response.json();
  if (!response.ok) {
    throw new Error(typeof body.detail === 'string' ? body.detail : 'Revisa los valores ingresados e intenta nuevamente.');
  }
  return body;
}

function drawScatter(model) {
  return ArriendoCharts.model(model, metadata);
}

function drawMAE() {
  return ArriendoCharts.overview(metadata);
}

function downloadCSV(filename, rows) {
  const quote = (v) => `"${String(v).replaceAll('"', '""')}"`;
  const csv = '\uFEFF' + rows.map((row) => row.map(quote).join(',')).join('\r\n');
  const url = URL.createObjectURL(new Blob([csv], {type: 'text/csv;charset=utf-8'}));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function populate(data) {
  if (!data.diagnostics || !window.Plotly) throw new Error('Faltan Plotly o los diagnósticos. Vuelve a ejecutar python -m arriendos.train.');
  metadata = data;
  data.comunas.forEach((comuna) => $('comuna').add(new Option(comuna, comuna)));
  if (data.comunas.includes('Santiago')) $('comuna').value = 'Santiago';
  Object.entries(data.models).forEach(([id, model]) => {
    const selected = id === data.selection.model;
    $('modelo').add(new Option(`${model.label}${selected ? ' · recomendado' : ''}`, id));
    $('chart-select').add(new Option(model.label, id));
    const row = document.createElement('tr');
    if (selected) row.className = 'selected';
    [model.label, money(model.test.mae), money(model.test.rmse), model.test.r2.toFixed(3)].forEach((value, i) => {
      const cell = document.createElement(i === 0 ? 'th' : 'td');
      if (i === 0) cell.scope = 'row';
      cell.textContent = value;
      if (selected && i === 0) {
        const badge = document.createElement('small');
        badge.textContent = 'Elegido por validación';
        cell.append(badge);
      }
      row.append(cell);
    });
    $('metrics-body').append(row);
  });
  $('modelo').value = data.selection.model;
  $('chart-select').value = data.selection.model;
  $('chart-select').disabled = false;
  $('download-metrics').disabled = false;
  $('download-predictions').disabled = false;
  $('stat-rows').textContent = data.data_audit.clean_rows.toLocaleString('es-CL');
  $('stat-test').textContent = data.split.rows.test.toLocaleString('es-CL');
  $('stat-model').textContent = data.models[data.selection.model].label;
  $('split-summary').textContent = `Filas: ${data.split.rows.train} de entrenamiento, ${data.split.rows.validation} de validación y ${data.split.rows.test} de prueba. Semilla: ${data.seed}.`;
  $('property-fields').disabled = false;
  $('load-status').hidden = true;
  const defaults = ['Santiago', 'Providencia', 'Las Condes'];
  for (let i = 1; i <= 3; i++) {
    const select = $(`compare-comuna-${i}`);
    data.comunas.forEach((c) => select.add(new Option(c, c)));
    select.value = data.comunas.includes(defaults[i - 1]) ? defaults[i - 1] : data.comunas[i - 1];
  }
  $('compare-fields').disabled = false;
  Promise.all([drawMAE(), drawScatter(data.selection.model), ArriendoCharts.initialize(data), runComparison()])
    .then(() => {window.chartsReady = true;})
    .catch((error) => {$('load-status').hidden = false; $('load-status').className = 'notice error'; $('load-status').textContent = `No se pudo completar el análisis: ${error.message}`;});
}

async function runComparison() {
  const revision = compareRevision;
  const comunas = [1, 2, 3].map((i) => $(`compare-comuna-${i}`).value);
  const propiedad = {comuna: comunas[0], metros: Number($('compare-metros').value),
    gastos_comunes: Number($('compare-gastos').value), habitaciones: Number($('compare-habitaciones').value),
    banos: Number($('compare-banos').value)};
  $('compare-button').disabled = true;
  $('compare-status').textContent = 'Calculando la misma propiedad en cada comuna…';
  try {
    await ArriendoCharts.communes(comunas);
    const result = await request('/api/compare', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({propiedad, comunas})});
    if (revision !== compareRevision) return;
    await ArriendoCharts.compare(result);
    $('compare-status').textContent = `${propiedad.metros} m² · gastos comunes ${money(propiedad.gastos_comunes)} · ${propiedad.habitaciones} habitaciones · ${propiedad.banos} baños. Las barras comparan los cuatro modelos y la referencia. Un hueco indica una predicción no válida.`;
  } catch (error) {
    if (revision !== compareRevision) return;
    Plotly.purge('communes-models');
    $('compare-status').textContent = `No se pudo comparar: ${error.message}`;
  } finally {
    $('compare-button').disabled = false;
  }
}
$('compare-form').addEventListener('submit', (event) => {event.preventDefault(); runComparison();});
$('compare-form').addEventListener('input', () => {
  compareRevision++;
  Plotly.purge('communes-models');
  $('compare-status').textContent = 'Características modificadas. Vuelve a comparar para actualizar los precios.';
});

$('prediction-form').addEventListener('input', () => {
  inputRevision++;
  $('price').textContent = '—';
  $('prediction-detail').textContent = 'Características modificadas. Vuelve a estimar para actualizar el resultado.';
  $('classification').textContent = '';
  $('warnings').replaceChildren();
  if (hasNeighbors) {ArriendoCharts.neighbors(); hasNeighbors = false;}
});
$('chart-select').addEventListener('change', () => drawScatter($('chart-select').value));
$('download-metrics').addEventListener('click', () => downloadCSV('comparacion-modelos.csv', [
  ['modelo', 'mae_prueba_clp', 'rmse_prueba_clp', 'r2_prueba', 'seleccionado_por_validacion'],
  ...Object.entries(metadata.models).map(([id, m]) => [m.label, m.test.mae, m.test.rmse, m.test.r2, id === metadata.selection.model]),
]));
$('download-predictions').addEventListener('click', () => {
  const id = $('chart-select').value;
  downloadCSV(`predicciones-${id}.csv`, [
    ['real_clp', 'estimado_clp', 'residuo_clp'],
    ...metadata.holdout.actual.map((actual, i) => {
      const predicted = metadata.holdout.predicted[id][i];
      return [actual, predicted, actual - predicted];
    }),
  ]);
});
$('prediction-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const revision = inputRevision;
  const data = {comuna: $('comuna').value, modelo: $('modelo').value};
  ['metros', 'gastos_comunes', 'habitaciones', 'banos'].forEach((key) => {data[key] = Number($(key).value);});
  $('predict-button').disabled = true;
  $('price').textContent = '…';
  $('prediction-detail').textContent = 'Consultando el modelo entrenado…';
  $('classification').textContent = '';
  $('warnings').replaceChildren();
  try {
    const result = await request('/api/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)});
    if (revision !== inputRevision) return;
    $('price').textContent = money(result.precio_clp);
    $('prediction-detail').textContent = `${metadata.models[result.modelo].label} · Error absoluto medio en prueba: ${money(result.mae_prueba_clp)}.`;
    $('classification').textContent = `Clasificación KNN independiente: ${result.clasificacion_knn.toLowerCase()} de ${money(result.umbral_clasificacion_clp)}. Puede diferir del precio calculado por el modelo seleccionado.`;
    await ArriendoCharts.neighbors(data, result.vecinos_knn);
    hasNeighbors = true;
    result.advertencias.forEach((message) => {const li = document.createElement('li'); li.textContent = message; $('warnings').append(li);});
  } catch (error) {
    if (revision !== inputRevision) return;
    $('price').textContent = '—';
    $('prediction-detail').textContent = `No se pudo calcular: ${error.message}`;
  } finally {
    $('predict-button').disabled = false;
  }
});
request('/api/metadata').then(populate).catch((error) => {
  $('load-status').className = 'notice error';
  $('load-status').textContent = `No se pudieron cargar los modelos. ${error.message}`;
});
