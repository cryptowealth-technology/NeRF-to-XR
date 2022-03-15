/**
 * The a global dictionary containing scene parameters.
 * @type {?Object}
 */
window.gSceneParams = null;

/**
 * The timestamp of the last frame to be rendered, used to track performance.
 * @type {number}
 */
window.gLastFrame = window.performance.now();

/**
 * The near plane used for rendering. Increasing this value somewhat speeds up
 * rendering, but this is most useful to show cross sections of the scene.
 * @type {number}
 */
window.gNearPlane = 0.33;

/**
 * This scene renders the baked NeRF reconstruction using ray marching.
 * @type {?THREE.Scene}
 */
window.gRayMarchScene = null;

/**
 * Progress counters for loading RGBA textures.
 * @type {number}
 */
window.gLoadedRGBATextures = 0;

/**
 * Progress counters for loading feature textures.
 * @type {number}
 */
window.gLoadedFeatureTextures = 0;

/**
 * Number of textures to load.
 * @type {number}
 */
window.gNumTextures = 0;

/**
 * The THREE.js renderer object we use.
 * @type {?THREE.WebGLRenderer}
 */
window.gRenderer = null;

/**
 * The perspective camera we use to view the scene.
 * @type {?THREE.PerspectiveCamera}
 */
window.gCamera = null;

/**
 * We control the perspective camera above using OrbitControls.
 * @type {?THREE.OrbitControls}
 */
window.gOrbitControls = null;

/**
 * An orthographic camera used to kick off ray marching with a
 * full-screen render pass.
 * @type {?THREE.OrthographicCamera}
 */
window.gBlitCamera = null;

/**
 * Reports an error to the user by populating the error div with text.
 * @param {string} text
 */
function error(text) {
  const e = document.getElementById('error');
  e.textContent = text;
  e.style.display = 'block';
}

/**
 * Creates a DOM element that belongs to the given CSS class.
 * @param {string} what
 * @param {string} classname
 * @return {!HTMLElement}
 */
function create(what, classname) {
  const e = /** @type {!HTMLElement} */(document.createElement(what));
  if (classname) {
    e.className = classname;
  }
  return e;
}


/**
 * Resizes a DOM element to the given dimensions.
 * @param {!Element} element
 * @param {number} width
 * @param {number} height
 */
function setDims(element, width, height) {
  element.style.width = width.toFixed(2) + 'px';
  element.style.height = height.toFixed(2) + 'px';
}

/**
 * Hides the Loading prompt.
 */
function hideLoading() {
  let loading = document.getElementById('Loading');
  loading.style.display = 'none';

  let loadingContainer = document.getElementById('loading-container');
  loadingContainer.style.display = 'none';
}

/**
 * Updates the loading progress HTML elements.
 */
function updateLoadingProgress() {
  let texturergbprogress = document.getElementById('texturergbprogress');
  let texturefeaturesprogress =
      document.getElementById('texturefeaturesprogress');

  const textureString = gNumTextures > 0 ? gNumTextures : '?';
  texturergbprogress.innerHTML =
      'RGBA images: ' + gLoadedRGBATextures + '/' + textureString;
  texturefeaturesprogress.innerHTML =
      'feature images: ' + gLoadedFeatureTextures + '/' + textureString;
}


/**
 * Checks whether the WebGL context is valid and the underlying hardware is
 * powerful enough. Otherwise displays a warning.
 * @return {boolean}
 */
function isRendererUnsupported() {
  let loading = document.getElementById('Loading');

  let gl = document.getElementsByTagName("canvas")[0].getContext('webgl2');
  if (!gl) {
    loading.innerHTML = "Error: WebGL2 context not found. Is your machine" +
    " equipped with a discrete GPU?";
    return true;
  }

  let debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
  if (!debugInfo) {
    loading.innerHTML = "Error: Could not fetch renderer info. Is your" +
    " machine equipped with a discrete GPU?";
    return true;
  }

  let renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
  if (!renderer || renderer.search("SwiftShader") >= 0 ||
      (renderer.search("ANGLE") >= 0 &&
       renderer.search("Intel") >= 0 &&
       (renderer.search("HD Graphics") >= 0 ||
        renderer.search("UHD Graphics") >= 0))) {
  loading.innerHTML = "Error: Unsupported renderer: " + renderer +
    ". Are you running with hardware acceleration enabled?";
    return true;
  }

  return false;
}


/**
 * Updates the frame rate counter using exponential fall-off smoothing.
 */
function updateFPSCounter() {
  let currentFrame = window.performance.now();
  let milliseconds = currentFrame - gLastFrame;
  let oldMilliseconds = 1000 /
      (parseFloat(document.getElementById('fpsdisplay').innerHTML) || 1.0);

  // Prevent the FPS from getting stuck by ignoring frame times over 2 seconds.
  if (oldMilliseconds > 2000 || oldMilliseconds < 0) {
    oldMilliseconds = milliseconds;
  }
  let smoothMilliseconds = oldMilliseconds * (0.75) + milliseconds * 0.25;
  let smoothFps = 1000 / smoothMilliseconds;
  gLastFrame = currentFrame;
  document.getElementById('fpsdisplay').innerHTML = smoothFps.toFixed(1);

  // Let's store this FPS (for benchmarking the first 60 s)
  window.fpsValuesOfCanvas.push(smoothFps)
}