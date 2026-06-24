# TODO — dexjoco-shadow (teleop Shadow en DexJoCo)

Estado: branch `shadow-support` pusheada. Attach Panda+Shadow funciona, escena
bucket carga y renderiza (nq=46). Falta hacerla controlable + teleoperable.

## Hecho
- [x] Attach Panda+Shadow vía MjSpec (`shadow_ext/build.py`).
- [x] Shadow Hand vendorizada (`sim/envs/xmls/shadow_hand/`, de Menagerie).
- [x] Escena bucket con Shadow compila y se ve en viewer.
- [x] Commit + push a fork (branch `shadow-support`).

## Siguiente
- [ ] **Env Shadow**: editar `panda_pick_bucket_env.py` (o copia) →
      `_ALLEGRO_JOINT_NAMES`→Shadow, `_N_ALLEGRO` 16→20, ctrl ids, fingertip bodies.
- [ ] **Mapeo qpos→ctrl**: Shadow 24 juntas → 20 actuadores (tendones acoplados).
      Confirmar cómo el retargeter (24 qpos, orden Menagerie) entra al ctrl.
- [ ] **OSC**: arrancar para que el brazo se sostenga y obedezca target de muñeca.
      Re-tune kp (Shadow pesa más que Allegro).
- [ ] **Teleop UDP**: `tasks/sim_teleop.py` puertos 5012 (muñeca) / 5014 (mano).
      Emitir desde nuestro pipeline: muñeca WiLoR → 5012, qpos retargeter → 5014.
- [ ] **Muñeca WiLoR**: surfacear pose global en `wilor_source.py` (hoy se tira al
      canonicalizar a frame Dong). pos = keypoint[0] cámara; rot = frame Dong.
      Filtro one-euro o anclaje (señal monocular tiembla + ~253ms lag).

## Plan por etapas (riesgo creciente, cada una ya es demo)
1. [ ] Muñeca fija + dedos en vivo (objeto pre-colocado, cerrar → agarrar).
2. [ ] + orientación muñeca en vivo (frame Dong, confiable).
3. [ ] + traslación muñeca en vivo (cam_t + filtro). Si tiembla mucho, quedarse en 2.

## Notas
- Ref completa de hallazgos/diseño: `AIST-hand/docs/sim_eval_findings.md`.
- Eval cinemático (RS/NDS/NVS/S_k sobre test split) = núcleo separado, ambos robots,
  sin sim. Esto (DexJoCo+Shadow) = capa funcional.
- Allegro ya está soportado nativo en DexJoCo (no requiere attach).
- Robar de DexGraspBench `MjHO` (mano flotante mocap + física) si se prefiere
  demo sin brazo en vez de OSC. AnyTeleop = referencia de diseño (SAPIEN, no correr).
