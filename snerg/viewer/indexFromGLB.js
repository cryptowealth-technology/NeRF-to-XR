/**
 * Loads the initial SNeRG scene parameters.
 * @param {string} dirUrl
 * @param {number} width
 * @param {number} height
 * @returns {!Promise} A promise for when the initial scene params have loaded.
 */
function loadSceneGLB(dirUrl, width, height) {
  // Reset the texture loading window.
  gLoadedRGBATextures = gLoadedFeatureTextures = gNumTextures = 0;
  // updateLoadingProgress();

  // Loads scene parameters (voxel grid size, NDC/no-NDC, view-dependence MLP).
  let modelResourceUrl = dirUrl + '/' + 'engine_default.glb';
  // Instantiate a loader, that utilizes Draco compression
  const loader = new THREE.GLTFLoader();
    const dracoLoader = new THREE.DRACOLoader();
    dracoLoader.setDecoderPath('https://unpkg.com/three@0.113.1/examples/js/libs/draco/')
    loader.setDRACOLoader( dracoLoader );
  // Load a glTF resource
  loader.load(
    // resource URL
    modelResourceUrl,
    // called when the resource is loaded
    function ( gltf ) {
      gRayMarchScene = new THREE.Scene();
      gRayMarchScene.add( gltf.scene );
      // gRayMarchScene.autoUpdate = false;
      // (6) add lights and camera
      const light = new THREE.DirectionalLight(0xFFFFFF, 1);  // white light, intensity = 1
        light.position.set(-1, 2, 4);
        gRayMarchScene.add(light);
      gBlitCamera = new THREE.OrthographicCamera(
          width / -2, width / 2, height / 2, height / -2, -10000, 10000);
      gBlitCamera.position.z = 10;
      requestAnimationFrame(renderGLB);
      console.log("Model has been loaded!")
    },
    // called while loading is progressing
    function ( xhr ) {},
    // called when loading has errors
    function ( errors ) {
      console.error(
        'Could not load scene from: ' + dirUrl + ', errors:\n\t' + errors);
    }
  );    
}

/**
 * Initializes the application based on the URL parameters.
 */
function initFromParametersGLB() {
  const params = new URL(window.location.href).searchParams;
  const dirUrl = params.get('dir');
  const size = params.get('s');

  const usageString =
      'To view a SNeRG scene, specify the following parameters in the URL:\n' +
      '(Required) The URL to a SNeRG scene directory.\n' +
      's: (Optional) The dimensions as width,height. E.g. 640,360.\n' +
      'vfovy:  (Optional) The vertical field of view of the viewer.';

  if (!dirUrl) {
    error('dir is a required parameter.\n\n' + usageString);
    return;
  }
  
  // Set size of canvas where model is seen
  let width = 1280;
  let height = 720;
  if (size) {
    const match = size.match(/([\d]+),([\d]+)/);
    width = parseInt(match[1], 10);
    height = parseInt(match[2], 10);
  }

  gNearPlane = parseFloat(params.get('near') || 0.33);
  const vfovy = parseFloat(params.get('vfovy') || 35);


  const view = create('div', 'view');
  setDims(view, width, height);
  view.textContent = '';

  const viewSpaceContainer = document.getElementById('viewspacecontainer');
  viewSpaceContainer.style.display = 'inline-block';

  const viewSpace = document.querySelector('.viewspace');
  viewSpace.textContent = '';
  viewSpace.appendChild(view);
  
  // (1) get canvas for the viewer
  let canvas = document.createElement('canvas');
  view.appendChild(canvas);

  // (2) Set up a high performance WebGL context, making sure that anti-aliasing is
  // turned off.
  let gl = canvas.getContext('webgl2');
  gRenderer = new THREE.WebGLRenderer({
    canvas: canvas,
    context: gl,
    powerPreference: 'low-power',
    alpha: false,
    stencil: false,
    precision: 'mediump',
    depth: false,
    antialias: false,
    desynchronized: true
  });

  // (3) init camera
  gCamera = new THREE.PerspectiveCamera(
      72, canvas.offsetWidth / canvas.offsetHeight, gNearPlane, 100.0);
  gCamera.aspect = view.offsetWidth / view.offsetHeight;
  gCamera.fov = vfovy;
  gRenderer.autoClear = false;
  gRenderer.setSize(view.offsetWidth, view.offsetHeight);

  // (4) init orbit controls
  gOrbitControls = new THREE.OrbitControls(gCamera, view);
  gOrbitControls.screenSpacePanning = true;
  gOrbitControls.zoomSpeed = 0.5;

  // (5) init scene and models
  loadSceneGLB(dirUrl, width, height);
}


/**
 * Set up code that needs to run once after the  scene parameters have loaded.
 */
function loadOnFirstFrameGLB() {
  // Early out if we've already run this function.
  if (gSceneParams['hasRunBefore']) {
    return;
  }

  // Set up the camera controls for the scene type.
  gOrbitControls.target.x = 0.0;
  gOrbitControls.target.y = 0.0;
  gOrbitControls.target.z = 0.0;
  gCamera.position.set(0.0, 0, 5.0);
  gOrbitControls.update();
  // gCamera.updateProjectionMatrix();
  gOrbitControls.update();

  hideLoading();

  // Now set the loading textures flag so this function runs only once.
  gSceneParams['hasRunBefore'] = true;
}


/**
 * The main render function that gets called every frame.
 * @param {number} t
 */
function renderGLB(t) {
  loadOnFirstFrameGLB();

  gOrbitControls.update();
  gRenderer.render(gRayMarchScene, gCamera);  // b/c no ray marching used, we can just use the perspective camera

  updateFPSCounter();
  requestAnimationFrame(renderGLB);
}

/**
 * Starts the volumetric object viewer application.
 */
function startFromGLB() {
  // init array to store FPS values
  window.fpsValuesOfCanvas = new Array();
  // build the viewer
  gSceneParams = {};
  initFromParametersGLB();
  // addHandlers();
  // Start rendering 
  // store FPS times over first 60 s
  setTimeout(() => {
      console.log(window.fpsValuesOfCanvas)
    }, 60*1000, 
  )
}

// startFromGLB();
