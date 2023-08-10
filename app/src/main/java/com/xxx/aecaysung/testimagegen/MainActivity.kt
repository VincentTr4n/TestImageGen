package com.xxx.aecaysung.testimagegen

import android.graphics.Bitmap
import android.os.Bundle
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.contrib.android.TensorFlowInferenceInterface
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader

class MainActivity : AppCompatActivity() {

    private var outputNodes: Array<String>? = null
    private var WORD_MAP: Array<String>? = null
    private val inferenceInterface by lazy { initSession() }
    private lateinit var pickMedia: ActivityResultLauncher<PickVisualMediaRequest>
    private val imageView: ImageView by lazy { findViewById(R.id.imageView) }
    private val textView: TextView by lazy { findViewById(R.id.txt_caption) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        pickMedia = registerForActivityResult(PickVisualMedia()) { uri ->
            // Callback is invoked after the user selects a media item or closes the
            // photo picker.
            if (uri != null) {
                Log.d("tran.dc", "Selected URI: $uri")
                lifecycleScope.launch {
                    val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)
                    imageView.setImageBitmap(bitmap)
                    val text = withContext(Dispatchers.IO) {
                        generateCaptions(preprocess(bitmap))
                    }
                    Log.d("tran.dc", "thread - ${Thread.currentThread()}")
                    textView.text = text
                }
            } else {
                Log.d("tran.dc", "No media selected")
            }
        }
        Log.d("tran.dc", "inferenceInterface: $inferenceInterface")

        findViewById<Button>(R.id.imagePicker).setOnClickListener {
            pickMedia.launch(PickVisualMediaRequest(PickVisualMedia.ImageOnly))
        }
    }

    private fun initSession(): TensorFlowInferenceInterface {
        outputNodes = loadFile(OUTPUT_NODES)
        WORD_MAP = loadFile("idmap")
        Log.d("tran.dc", "outputNodes: \n${outputNodes?.joinToString("\n") { "#$it#" }}")
        return TensorFlowInferenceInterface(assets, MODEL_FILE)
    }

    private fun loadFile(fileName: String): Array<String> {
        var ins: InputStream? = null
        try {
            ins = assets.open(fileName)
        } catch (e: IOException) {
            e.printStackTrace()
        }
        val r = BufferedReader(InputStreamReader(ins))
        val total = StringBuilder()
        var line: String?
        try {
            while (r.readLine().also { line = it } != null) {
                total.append(line).append('\n')
            }
        } catch (e: IOException) {
            e.printStackTrace()
        }
        if (total.last() == '\n') total.deleteCharAt(total.length - 1);
        return total.toString().split("\n").toTypedArray()
    }

    fun preprocess(srcImage: Bitmap): FloatArray? {
        val imBitmap = Bitmap.createScaledBitmap(srcImage, IMAGE_SIZE, IMAGE_SIZE, true)
        val intValues = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        val floatValues = FloatArray(IMAGE_SIZE * IMAGE_SIZE * 3)
        imBitmap.getPixels(intValues, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
        for (i in intValues.indices) {
            val value = intValues[i]
            floatValues[i * 3] = (value shr 16 and 0xFF).toFloat() / 255 //R
            floatValues[i * 3 + 1] = (value shr 8 and 0xFF).toFloat() / 255 //G
            floatValues[i * 3 + 2] = (value and 0xFF).toFloat() / 255 //B
        }
        return floatValues
    }

    fun generateCaptions(imRGBMatrix: FloatArray?): String {
        val startTime = SystemClock.currentThreadTimeMillis()
        inferenceInterface.feed(
            INPUT1,
            imRGBMatrix,
            DIM_IMAGE[0].toLong(),
            DIM_IMAGE[1].toLong(),
            DIM_IMAGE[2].toLong(),
            DIM_IMAGE[3].toLong()
        )
        inferenceInterface.run(outputNodes)
        var result = ""
        val temp = Array(NUM_TIMESTEPS) { IntArray(1) }
        for (i in 0 until NUM_TIMESTEPS) {
            inferenceInterface.fetch(outputNodes!![i], temp[i])
            if (temp[i][0] == 2 /*</S>*/) {
                val costTime = SystemClock.currentThreadTimeMillis() - startTime
                Log.i("tran.dc", "GenerateCaptions end, cost time=" + costTime + "ms")
                return result
            }
            result += WORD_MAP!![temp[i][0]] + " "
        }
        return "NULL"
    }

    companion object {
        private const val MODEL_FILE = "file:///android_asset/merged_frozen_graph.pb"
        private const val INPUT1 = "encoder/import/input:0"
        private const val OUTPUT_NODES = "DecoderOutputs.txt"
        private const val NUM_TIMESTEPS = 22
        private const val IMAGE_SIZE = 224
        private const val IMAGE_CHANNELS = 3
        private val DIM_IMAGE = intArrayOf(1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
    }
}