import * as THREE from 'three';
import { DensityField } from './DensityField';
import vertexShader from './shaders/water.vert.glsl?raw';
import fragmentShader from './shaders/water.frag.glsl?raw';

export class WaterRenderer {
  private densityField: DensityField;
  private texture: THREE.Data3DTexture;
  private material: THREE.ShaderMaterial;
  private mesh: THREE.Mesh;

  constructor(
    scene: THREE.Scene,
    domainMin: THREE.Vector3,
    domainMax: THREE.Vector3,
    options?: { resolution?: number; splatRadius?: number; threshold?: number },
  ) {
    const res = options?.resolution ?? 64;
    const splatRadius = options?.splatRadius ?? 0.1;

    const texData = new Float32Array(res * res * res * 2);
    this.texture = new THREE.Data3DTexture(texData, res, res, res);
    this.texture.format = THREE.RGFormat;
    this.texture.type = THREE.FloatType;
    this.texture.minFilter = THREE.LinearFilter;
    this.texture.magFilter = THREE.LinearFilter;
    this.texture.wrapS = THREE.ClampToEdgeWrapping;
    this.texture.wrapT = THREE.ClampToEdgeWrapping;
    this.texture.wrapR = THREE.ClampToEdgeWrapping;
    this.texture.needsUpdate = true;

    this.densityField = new DensityField(
      res,
      [domainMin.x, domainMin.y, domainMin.z],
      [domainMax.x, domainMax.y, domainMax.z],
      splatRadius,
      texData,
    );

    this.material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        uDensityTex: { value: this.texture },
        uDomainMin: { value: domainMin.clone() },
        uDomainMax: { value: domainMax.clone() },
        uLightDir: { value: new THREE.Vector3() },
        uLightColor: { value: new THREE.Color(1, 1, 1) },
        uThreshold: { value: options?.threshold ?? 0.5 },
        uBgColor: { value: new THREE.Color(0xd0d8e8) },
        uInvViewProjection: { value: new THREE.Matrix4() },
        uResolution: { value: new THREE.Vector2() },
      },
      side: THREE.BackSide,
      transparent: false,
      depthWrite: true,
    });

    const size = new THREE.Vector3().subVectors(domainMax, domainMin);
    const center = new THREE.Vector3().addVectors(domainMin, domainMax).multiplyScalar(0.5);
    const boxGeo = new THREE.BoxGeometry(size.x, size.y, size.z, 1, 1, 1);
    this.mesh = new THREE.Mesh(boxGeo, this.material);
    this.mesh.position.copy(center);
    scene.add(this.mesh);
  }

  update(
    posX: Float32Array, posY: Float32Array, posZ: Float32Array,
    velX: Float32Array, velY: Float32Array, velZ: Float32Array,
    count: number,
    light: THREE.DirectionalLight,
    camera: THREE.Camera,
    rendererSize: THREE.Vector2,
  ) {
    this.densityField.splatParticles(posX, posY, posZ, velX, velY, velZ, count);
    this.texture.needsUpdate = true;

    const lightDir = light.position.clone().normalize();
    this.material.uniforms.uLightDir.value.copy(lightDir);
    this.material.uniforms.uLightColor.value.copy(light.color);

    const invVP = this.material.uniforms.uInvViewProjection.value as THREE.Matrix4;
    invVP.copy(camera.projectionMatrix).multiply(camera.matrixWorldInverse).invert();
    this.material.uniforms.uResolution.value.copy(rendererSize);
  }

  dispose() {
    this.texture.dispose();
    this.material.dispose();
    this.mesh.geometry.dispose();
  }
}
