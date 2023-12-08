const express = require('express');
const router = express.Router();
const fileService = require('../services/file.service');

router.post('/upload', async (req, res) => {
 const result = await fileService.uploadFile(req.files.file);
 res.status(result.status).json(result.data);
});



module.exports = router;
