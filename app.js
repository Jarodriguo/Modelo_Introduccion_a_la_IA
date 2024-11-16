async function predict() {
    const fileInput = document.getElementById('imageUpload');
    const resultText = document.getElementById('resultText');

    if (!fileInput.files || fileInput.files.length === 0) {
        alert("Por favor, selecciona una imagen primero.");
        return;
    }

    // Mostrar mensaje de carga
    resultText.innerText = "Procesando imagen...";

    try {
        // Cargar modelo de TensorFlow.js desde la misma carpeta
        const model = await tf.loadLayersModel('model.json');
        console.log("Modelo cargado")

        // Leer la imagen seleccionada por el usuario
        const file = fileInput.files[0];
        const image = await readImage(file);

        // Asegurarse de que la imagen esté cargada correctamente
        if (!image) {
            resultText.innerText = "Error al cargar la imagen. Por favor, intenta con otra imagen.";
            return;
        }

        // Preprocesar la imagen
        const tensor = tf.browser.fromPixels(image)
            .resizeNearestNeighbor([50, 50]) // Tamaño de entrada esperado por el modelo
            .toFloat()
            .div(tf.scalar(255)) // Normalización
            .expandDims();

        // Hacer predicción
        const prediction = model.predict(tensor);
        const predictionData = await prediction.data();

        // Mostrar el resultado
        resultText.innerText = predictionData[0] > 0.5 ? "Posible presencia de cáncer" : "Sin indicios de cáncer";

        // Limpiar el tensor
        tensor.dispose();
    } catch (error) {
        console.error("Error en la predicción:", error);
        resultText.innerText = "Error en la predicción. Por favor, inténtalo nuevamente.";
    }
}

// Función para leer la imagen como HTMLImageElement
function readImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const img = new Image();
            img.src = reader.result;
            img.onload = () => resolve(img);
            img.onerror = () => reject("No se pudo cargar la imagen.");
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}