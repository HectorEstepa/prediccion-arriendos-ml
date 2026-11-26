'use strict';
window.ArriendoCharts = (() => {
  const colors = ['#417c9b', '#a54730', '#678858', '#8c6bb1', '#999993'];
  const names = {metros: 'Superficie', gastos_comunes: 'Gastos comunes', habitaciones: 'Habitaciones', banos: 'Baños', comuna: 'Comuna'};
  let report;
  const currency = (v) => new Intl.NumberFormat('es-CL', {style: 'currency', currency: 'CLP', maximumFractionDigits: 0}).format(v);
  const config = {responsive: true, displaylogo: false, scrollZoom: false,
    modeBarButtonsToRemove: ['select2d', 'lasso2d'], toImageButtonOptions: {format: 'png', scale: 2}};
  const axis = (title) => ({title: {text: title}, automargin: true, gridcolor: '#e8edf1', zerolinecolor: '#a9b7c1'});
  function plot(id, traces, layout = {}) {
    return Plotly.react(id, traces, {
      autosize: true, margin: {l: 65, r: 22, t: 20, b: 65},
      font: {family: 'Segoe UI, Arial, sans-serif', size: 11, color: '#425464'},
      paper_bgcolor: '#fff', plot_bgcolor: '#fff', separators: ',.',
      colorway: colors, showlegend: false, hovermode: 'closest',
      xaxis: axis(''), yaxis: axis(''), ...layout,
    }, config);
  }
  function horizontal(id, rows, label, value, title, extra = {}) {
    return plot(id, [{type: 'bar', orientation: 'h', y: rows.map(label), x: rows.map(value),
      marker: {color: colors[0]}, hovertemplate: '%{y}<br>%{x:,.2f}<extra></extra>'}], {
      xaxis: axis(title), yaxis: {...axis(''), autorange: 'reversed', showgrid: false},
      margin: {l: 20, r: 25, t: 15, b: 60}, ...extra,
    });
  }
  function overview(data) {
    report = data;
    const entries = Object.entries(data.models).sort((a, b) => b[1].test.mae - a[1].test.mae);
    return plot('mae-chart', [
      {type: 'bar', orientation: 'h', name: 'Validación', y: entries.map(([, m]) => m.label),
        x: entries.map(([, m]) => m.validation.mae), marker: {color: '#b8cbd5'}, hovertemplate: '%{y}<br>Validación: $%{x:,.0f}<extra></extra>'},
      {type: 'bar', orientation: 'h', name: 'Prueba', y: entries.map(([, m]) => m.label),
        x: entries.map(([, m]) => m.test.mae), marker: {color: entries.map(([id]) => id === data.selection.model ? colors[1] : colors[0])},
        hovertemplate: '%{y}<br>Prueba: $%{x:,.0f}<extra></extra>'},
    ], {barmode: 'group', showlegend: true, legend: {orientation: 'h', y: 1.15},
      xaxis: {...axis('MAE (CLP)'), rangemode: 'tozero'},
      yaxis: {...axis(''), showgrid: false}, margin: {l: 10, r: 20, t: 30, b: 55}});
  }
  function model(id, data) {
    report = data;
    const actual = data.holdout.actual, predicted = data.holdout.predicted[id];
    const low = Math.min(0, ...predicted), high = Math.max(...actual, ...predicted);
    const work = [];
    work.push(plot('scatter', [
      {type: 'scatter', mode: 'markers', x: actual, y: predicted, marker: {size: 5, opacity: .55, color: colors[0]},
        hovertemplate: 'Real: $%{x:,.0f}<br>Estimado: $%{y:,.0f}<extra></extra>'},
      {type: 'scatter', mode: 'lines', x: [low, high], y: [low, high], line: {dash: 'dash', color: '#888'}, hoverinfo: 'skip'},
    ], {xaxis: axis('Precio real (CLP)'), yaxis: axis('Precio estimado (CLP)')}));
    work.push(plot('residuals', [{type: 'scatter', mode: 'markers', x: predicted,
      y: actual.map((v, i) => v - predicted[i]), marker: {color: colors[0], opacity: .55, size: 5},
      hovertemplate: 'Estimado: $%{x:,.0f}<br>Residuo: $%{y:,.0f}<extra></extra>'}],
    {xaxis: axis('Precio estimado (CLP)'), yaxis: axis('Real − estimado (CLP)'),
      shapes: [{type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0, y1: 0, line: {dash: 'dash', color: '#777'}}]}));
    const permutation = data.diagnostics.models[id].permutation || [];
    work.push(plot('permutation', [{type: 'bar', orientation: 'h',
      y: permutation.map((p) => names[p.feature]), x: permutation.map((p) => p.mean),
      error_x: {type: 'data', array: permutation.map((p) => p.std), visible: true}, marker: {color: colors[0]},
      hovertemplate: '%{y}<br>Incremento del MAE: $%{x:,.0f}<extra></extra>'}],
    {xaxis: axis('Incremento del MAE (CLP)'), yaxis: {...axis(''), autorange: 'reversed', showgrid: false},
      annotations: permutation.length ? [] : [{text: 'La referencia predice una mediana fija y no usa las características.', xref: 'paper', yref: 'paper', x: .5, y: .5, showarrow: false}]}));
    document.getElementById('chart-model').textContent = data.models[id].label;
    document.getElementById('chart-summary').textContent = `Fuente: evaluación del proyecto. ${actual.length} anuncios de prueba · MAE: ${currency(data.models[id].test.mae)}.`;
    const descriptions = {
      baseline: 'Predice la mediana del arriendo de entrenamiento y validación para todas las propiedades. Sirve como referencia mínima de desempeño.',
      lineal: 'Suma contribuciones lineales de las características y de la comuna. Se utiliza una comuna de referencia para evitar columnas redundantes. Los coeficientes se muestran más abajo en unidades originales.',
      knn: 'Busca cinco anuncios cercanos usando características estandarizadas y comuna codificada. Estima el arriendo mediante un promedio ponderado por la distancia. La clasificación alto/bajo es un experimento separado.',
      random_forest: 'Promedia 200 árboles de decisión, con al menos tres anuncios por hoja. Captura relaciones no lineales e interacciones entre características. Sus importancias no expresan la dirección del efecto.',
      red_neuronal: 'Red con capas ocultas de 64 y 32 unidades y activación ReLU. Estandariza las entradas y el precio objetivo. Utiliza parada temprana y un máximo de 600 iteraciones.',
    };
    document.getElementById('model-description').textContent = descriptions[id];
    const m = data.models[id];
    document.getElementById('model-metrics').textContent = `MAE de validación: ${currency(m.validation.mae)}. MAE de prueba: ${currency(m.test.mae)}. R² de prueba: ${m.test.r2.toFixed(3)}.`;
    return Promise.all(work);
  }
  function communes(selected) {
    const groups = selected.map((c) => report.diagnostics.communes.find((g) => g.comuna === c)).filter(Boolean);
    return plot('communes-box', groups.map((g, i) => ({type: 'box', name: g.comuna, y: g.prices,
      boxpoints: 'outliers', marker: {color: colors[i]}, hovertemplate: '$%{y:,.0f}<extra>%{fullData.name}</extra>'})),
    {yaxis: axis('Arriendo observado (CLP)'), xaxis: axis('Comuna')});
  }
  function compare(result) {
    return plot('communes-models', Object.entries(result.predicciones).map(([id, values], i) => ({
      type: 'bar', name: report.models[id].label, x: result.comunas, y: values,
      marker: {color: colors[i]}, hovertemplate: '%{x}<br>$%{y:,.0f}<extra>%{fullData.name}</extra>',
    })), {barmode: 'group', showlegend: true, legend: {orientation: 'h', y: -.3},
      xaxis: axis('Comuna'), yaxis: {...axis('Arriendo estimado (CLP)'), rangemode: 'tozero'},
      margin: {l: 65, r: 15, t: 15, b: 120}});
  }
  function neighbors(property, values = []) {
    const projection = report.diagnostics.knn_projection;
    const traces = [0, 1].map((label) => {
      const idx = projection.clase.map((v, i) => v === label ? i : -1).filter((i) => i >= 0);
      return {type: 'scatter', mode: 'markers', name: report.classification_knn.labels[label],
        x: idx.map((i) => projection.metros[i]), y: idx.map((i) => projection.gastos_comunes[i]),
        marker: {size: 4, opacity: .3, color: colors[label]},
        hovertemplate: '%{x} m²<br>Gastos: $%{y:,.0f}<extra>%{fullData.name}</extra>'};
    });
    if (property) {
      traces.push({type: 'scatter', mode: 'markers', name: 'Cinco vecinos', x: values.map((v) => v.metros),
        y: values.map((v) => v.gastos_comunes), text: values.map((v) => `${v.comuna} · arriendo ${currency(v.precio_clp)} · distancia ${v.distancia.toFixed(3)}`),
        marker: {size: 11, color: '#678858', symbol: 'circle-open', line: {width: 2}},
        hovertemplate: '%{text}<extra></extra>'});
      traces.push({type: 'scatter', mode: 'markers', name: 'Propiedad consultada', x: [property.metros], y: [property.gastos_comunes],
        marker: {symbol: 'star', size: 14, color: '#222'}, hovertemplate: '%{x} m²<br>Gastos: $%{y:,.0f}<extra>Consulta</extra>'});
    }
    return plot('knn-neighbors', traces, {showlegend: true, legend: {orientation: 'h', y: -.25},
      xaxis: axis('Superficie (m²)'), yaxis: axis('Gastos comunes (CLP)'), margin: {l: 65, r: 20, t: 15, b: 100}});
  }
  function initialize(data) {
    report = data;
    const d = data.diagnostics, jobs = [];
    const rank = (id, rows) => plot(id, [{type: 'bar', orientation: 'h', y: rows.map((r) => r.comuna),
      x: rows.map((r) => r.median), customdata: rows.map((r) => r.count), marker: {color: colors[0]},
      hovertemplate: '%{y}<br>Mediana: $%{x:,.0f}<br>Anuncios: %{customdata}<extra></extra>'}],
    {xaxis: axis('Mediana de arriendo (CLP)'), yaxis: {...axis(''), autorange: 'reversed', showgrid: false}, margin: {l: 20, r: 25, t: 15, b: 60}});
    jobs.push(rank('communes-high', d.communes.slice(0, 10)));
    jobs.push(rank('communes-low', [...d.communes].reverse().slice(0, 10)));
    const linear = d.models.lineal;
    document.getElementById('linear-reference').textContent = `Comuna de referencia: ${linear.reference_comuna}. Intercepto en unidades originales: ${currency(linear.intercept_clp)}; no equivale al arriendo de una propiedad real sin características.`;
    const units = {metros: 'Superficie · por m²', gastos_comunes: 'Gastos · por peso', habitaciones: 'Por habitación', banos: 'Por baño'};
    jobs.push(horizontal('linear-numeric', Object.entries(linear.numeric_coefficients), (r) => units[r[0]], (r) => r[1], 'Cambio estimado del arriendo (CLP)'));
    const coefs = [...linear.comuna_coefficients, {comuna: linear.reference_comuna, coefficient: 0}].sort((a, b) => b.coefficient - a.coefficient);
    jobs.push(horizontal('linear-communes', coefs, (r) => r.comuna, (r) => r.coefficient, `Cambio frente a ${linear.reference_comuna} (CLP)`));
    const forest = d.models.random_forest;
    document.getElementById('forest-description').textContent = `${forest.trees} árboles. Se muestran importancias por reducción de impureza; pueden favorecer algunas variables y no son efectos causales. La columna de referencia ${forest.reference_comuna} se omite en la codificación.`;
    jobs.push(horizontal('forest-numeric', Object.entries(forest.numeric_importance), (r) => names[r[0]], (r) => r[1] * 100, 'Importancia (%)'));
    jobs.push(horizontal('forest-communes', [...forest.comuna_importance].sort((a, b) => b.importance - a.importance).slice(0, 10), (r) => r.comuna, (r) => r.importance * 100, 'Importancia (%)'));
    const neural = d.models.red_neuronal;
    jobs.push(plot('neural-loss', [{type: 'scatter', mode: 'lines', x: neural.loss_curve.map((_, i) => i + 1), y: neural.loss_curve,
      line: {color: colors[0]}, hovertemplate: 'Iteración %{x}<br>Pérdida %{y:.5f}<extra></extra>'}],
    {xaxis: axis('Iteración'), yaxis: axis('Pérdida de entrenamiento')}));
    jobs.push(plot('neural-architecture', [{type: 'scatter', mode: 'lines+markers+text',
      x: neural.layers.map((_, i) => i), y: neural.layers.map(() => 0),
      text: neural.layers.map((n) => `${n} ${n === 1 ? 'unidad' : 'unidades'}`), textposition: 'top center',
      marker: {size: 22, color: colors[0]}, line: {color: '#aaa'}, hoverinfo: 'text'}],
    {xaxis: {...axis(''), tickvals: [0, 1, 2, 3], ticktext: ['Entrada', 'Oculta 1', 'Oculta 2', 'Salida'], range: [-.4, 3.4], showgrid: false, zeroline: false},
      yaxis: {visible: false, range: [-1, 1]}}));
    const c = data.classification_knn;
    document.getElementById('knn-description').textContent = `K=5, con votos ponderados por distancia. Dos clases: arriendo menor a ${currency(c.threshold_clp)}, o desde ese monto. El umbral se calcula con entrenamiento y validación; las métricas siguientes usan exclusivamente prueba.`;
    const scores = document.getElementById('knn-scores');
    scores.replaceChildren();
    [['Exactitud', c.accuracy], ['Precisión', c.precision], ['Sensibilidad', c.recall], ['F1', c.f1]].forEach(([label, value]) => {
      const p = document.createElement('p'); p.textContent = `${label}: ${(value * 100).toFixed(1)}%`; scores.append(p);
    });
    jobs.push(plot('knn-confusion', [{type: 'heatmap', x: c.labels, y: c.labels, z: c.confusion_matrix,
      text: c.confusion_matrix, texttemplate: '%{z}', colorscale: [[0, '#f1f4f5'], [1, '#417c9b']],
      hovertemplate: 'Real: %{y}<br>Predicho: %{x}<br>Anuncios: %{z}<extra></extra>'}],
    {xaxis: axis('Clase predicha'), yaxis: {...axis('Clase real'), autorange: 'reversed'}}));
    jobs.push(plot('knn-roc', c.roc ? [
      {type: 'scatter', mode: 'lines', name: `AUC = ${c.auc.toFixed(3)}`, x: c.roc.fpr, y: c.roc.tpr, line: {color: colors[0]}},
      {type: 'scatter', mode: 'lines', name: 'Referencia aleatoria', x: [0, 1], y: [0, 1], line: {dash: 'dash', color: '#999'}},
    ] : [], {showlegend: true, legend: {orientation: 'h', y: -.3}, xaxis: {...axis('Tasa de falsos positivos'), range: [0, 1]},
      yaxis: {...axis('Tasa de verdaderos positivos'), range: [0, 1]}, margin: {l: 60, r: 20, t: 15, b: 100}}));
    const k = d.k_analysis;
    jobs.push(plot('knn-k', [
      {type: 'scatter', mode: 'lines+markers', name: 'Exactitud', x: k.rows.map((r) => r.k), y: k.rows.map((r) => r.accuracy)},
      {type: 'scatter', mode: 'lines+markers', name: 'F1', x: k.rows.map((r) => r.k), y: k.rows.map((r) => r.f1)},
    ], {showlegend: true, legend: {orientation: 'h', y: 1.15}, xaxis: {...axis('Número de vecinos (K)'), dtick: 2},
      yaxis: {...axis('Métrica en validación'), range: [0, 1]},
      shapes: [{type: 'line', x0: 5, x1: 5, yref: 'paper', y0: 0, y1: 1, line: {color: '#aaa', dash: 'dot'}}]}));
    jobs.push(neighbors());
    return Promise.all(jobs);
  }
  return {overview, model, initialize, communes, compare, neighbors};
})();
