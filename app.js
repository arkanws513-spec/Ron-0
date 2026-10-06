(() => {
  const note = document.getElementById("note");
  const button = document.getElementById("start");
  button.addEventListener("click", () => {
    note.textContent = "رون يعمل الآن. تم تجهيز الواجهة كبداية آمنة للنواة.";
    button.textContent = "رون جاهز";
    button.disabled = true;
  });
})();