const canvas = document.getElementById("canvas_input");
const canvas_clear = document.getElementById("canvas_clear");
const canvas_undo = document.getElementById("canvas_undo");
const canvas_predict = document.getElementById("canvas_predict");

let context = canvas.getContext("2d");
let background_color = "white"
let draw_color = "black";
let draw_width = "12";
let is_drawing = false;
let restore_array = [];
let index = -1;

canvas.width = canvas.height = 200;

context.fillStyle = background_color;
context.fillRect(0, 0, canvas.width, canvas.height);

canvas.addEventListener("touchstart", start, false);
canvas.addEventListener("mousedown", start, false);
canvas.addEventListener("touchmove", draw, false);
canvas.addEventListener("mousemove", draw, false);
canvas.addEventListener("touchend", stop, false);
canvas.addEventListener("mouseup", stop, false);
canvas.addEventListener("mouseout", stop, false);
canvas_clear.addEventListener("click", canvasClear, false);
canvas_undo.addEventListener("click", canvasUndo, false);
canvas_predict.addEventListener("click", canvasPredict, false);

function change_color(element) {
    draw_color = element.style.background;
}

function start(event) {
    is_drawing = true;
    context.beginPath();
    context.moveTo(
        event.clientX - canvas.offsetLeft, 
        event.clientY - canvas.offsetTop
        );
    event.preventDefault();
}

function draw(event) {
    if(is_drawing) {
        context.lineTo(
            event.clientX - canvas.offsetLeft, 
            event.clientY - canvas.offsetTop
            );
        context.strokeStyle = draw_color;
        context.lineWidth = draw_width;
        context.lineCap = "round";
        context.lineJoin = "round";
        context.stroke();
    }
    event.preventDefault();
}

function stop(event) {
    if(is_drawing) {
        context.stroke();
        context.closePath();
        is_drawing = false;
    }
    event.preventDefault();

    if(event.type != "mouseout") {
        restore_array.push(context.getImageData(0, 0, canvas.width, canvas.height));
        index += 1;
    }
}

function canvasClear() {
    context.fillStyle = background_color;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillRect(0, 0, canvas.width, canvas.height);

    restore_array = [];
    index = -1;
}

function canvasUndo() {
    if(index <= 0) {
        canvasClear();
    } else {
        index -= 1;
        restore_array.pop();
        context.putImageData(restore_array[index], 0, 0);
    }
}

function canvasPredict(event) {
    const img = canvas.toDataURL();
    $.ajax({
        type: "POST",
        url: "/predict/",
        data: img,
        success: function(result){
            $('#NN_result').text('Forecast: ' + result["NN_result"]);
            $('#CNN_result').text('Forecast: ' + result["CNN_result"]);
            $('#NN2_result').text('Forecast: ' + result["NN2_result"]);
            $('#CNN2_result').text('Forecast: ' + result["CNN2_result"]);
            let NN_pred = JSON.parse(result["NN_pred"]);
            let NN_keys = Object.keys(NN_pred);
            let NN_values = Object.values(NN_pred);
            highcharts_bar(NN_keys, NN_values, "#NN_result_bar");
            let CNN_pred = JSON.parse(result["CNN_pred"]);
            let CNN_keys = Object.keys(CNN_pred);
            let CNN_values = Object.values(CNN_pred);
            highcharts_bar(CNN_keys, CNN_values, "#CNN_result_bar");
            let NN2_pred = JSON.parse(result["NN2_pred"]);
            let NN2_keys = Object.keys(NN2_pred);
            let NN2_values = Object.values(NN2_pred);
            highcharts_bar(NN2_keys, NN2_values, "#NN2_result_bar");
            let CNN2_pred = JSON.parse(result["CNN2_pred"]);
            let CNN2_keys = Object.keys(CNN2_pred);
            let CNN2_values = Object.values(CNN2_pred);
            highcharts_bar(CNN2_keys, CNN2_values, "#CNN2_result_bar");

            var canvas_output = document.getElementById("canvas_output");
            var process_output = document.getElementById("process_output");
            canvas_output.src = "./static/images/canvas_img.png?" + new Date().getTime();
            process_output.src = "./static/images/process_img.png?" + new Date().getTime();

        }
    });

    canvasClear();

    event.preventDefault();
}

function highcharts_bar(keys, values, id) {
    var chart = {
        type: "bar"
    };
    var title = {
        text: null  
    };
    var xAxis = {
        categories: keys,
        title :{
            text : "Labels"
        }
    };
    var yAxis = {
        title :{
            text : "Probability"
        }
    };
    var series =  [
    {
        name: 'Probability',
        data: values
    }
    ];
    var legend = { 
        enabled: false 
    };

    var json = {};
    json.chart = chart;
    json.title = title;
    json.xAxis = xAxis;
    json.yAxis = yAxis;
    json.series = series;
    json.legend = legend;

    return $(id).highcharts(json);
}

