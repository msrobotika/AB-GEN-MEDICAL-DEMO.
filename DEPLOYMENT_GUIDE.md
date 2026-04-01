# 🚀 Guía de Despliegue en GitHub: AB-GEN Medical Demo

Esta guía te explica paso a paso cómo subir esta demo a tu repositorio de GitHub de forma profesional, asegurando que el código sea funcional y que la propiedad intelectual esté protegida.

## 1. Preparar el Repositorio Local

Abre una terminal en la carpeta `AB-GEN_MEDICAL_DEMO` y ejecuta:

```bash
git init
git add .
git commit -m "Initial commit: AB-GEN Medical Demo - Trazabilidad XAI Full"
```

> [!IMPORTANT]
> El archivo **`.gitignore`** ya está configurado para NO subir los modelos pesados (`.pkl`, `.npz`). Esto es crucial para mantener el repositorio ligero y proteger tu lógica de entrenamiento.

## 2. Crear Repositorio en GitHub

1. Ve a [github.com/new](https://github.com/new).
2. Nombre: `AB-GEN-MEDICAL-DEMO` (o el que prefieras).
3. Visibilidad: **Público** (recomendado para la venta) o **Privado**.
4. **NO** marques las opciones de añadir README, .gitignore o licencia (ya los tenemos creados).
5. Haz clic en **Create repository**.

## 3. Vincular y Subir el Código

Copia los comandos que te da GitHub (especialmente el de `remote add`) y ejecútalos:

```bash
git remote add origin https://github.com/TU_USUARIO/AB-GEN-MEDICAL-DEMO.git
git branch -M main
git push -u origin main
```

## 4. Gestión de Archivos del Modelo (Crítico)

Tus modelos (`.pkl` y `.npz`) pesan unos 150MB. Tienes dos opciones para que otros puedan probar la demo:

### Opción A: Descarga Externa (Recomendada)
1. Sube los modelos a **Google Drive, Dropbox o Azure Storage**.
2. Pon el enlace de descarga en el `README.md` (donde dice "📦 Modelos Pre-entrenados").

### Opción B: Git LFS (Solo si quieres que estén en GitHub)
1. Instala [Git LFS](https://git-lfs.github.com/).
2. Ejecuta `git lfs track "*.pkl"` y `git lfs track "*.npz"`.
3. Sube los archivos (GitHub tiene límites de ancho de banda gratuito para LFS).

## 5. Pulir la Presentación

1. **Video**: Una vez subas el video a YouTube, edita el `README.md` de GitHub y reemplaza el placeholder con el enlace real.
2. **About**: En el panel derecho de GitHub, añade una descripción corta: *"IA Médica Trazable para Diagnóstico Hematológico con Mapas de Calor XAI y Eficiencia Green AI."*
3. **Topics**: Añade tags como `medical-ai`, `xai`, `explainable-ai`, `bloodmnist`, `inference-demo`.

---

**¡Listo!** Ya tienes un arsenal de venta profesional que cualquier inversor o cliente puede descargar y probar con un solo click.
