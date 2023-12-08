import { Component } from '@angular/core';
import { FileService } from '../file.service';

@Component({
 selector: 'app-upload-file',
 templateUrl: './upload-file.component.html',
 styleUrls: ['./upload-file.component.css']
})
export class UploadFileComponent {
 selectedFile: File;

 constructor(private fileService: FileService) {}

 onFileSelected(event) {
    this.selectedFile = event.target.files[0];
 }

 onUpload() {
    this.fileService.uploadFile(this.selectedFile);
 }
}
